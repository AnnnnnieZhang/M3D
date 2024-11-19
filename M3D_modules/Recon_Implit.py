class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            config,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
    ):
        super().__init__()

        self.use_global_encoder = config['model']['latent_feature']['use_global_encoder']
        self.use_cls_encoder = config['model']['latent_feature']['use_cls_encoder']

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn

            # use cat architecture
            dims[0] = input_ch + 256                                # 39(x position encoder) + 256(pixel align feature)
            if self.use_global_encoder:
                dims[0] = dims[0] + 256                             # + 256(global img feature)
            if self.use_cls_encoder:
                dims[0] = dims[0] + 9                               # + 9(cls feature)
            skip_dim = skip_in[0]                                   # only one skip
            dims[skip_dim] = dims[0]

        print(multires, dims)
        self.num_layers = len(dims)                             # 9
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):

            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, latent_feature, cat_feature):
        if self.embed_fn is not None:
            input = self.embed_fn(input)                                    # [B*N_ray*N_sample, 39]

        x = input                                                           # [B*N_ray*N_sample, 39]

        num_repeats = x.shape[0] // cat_feature.shape[0]
        cat_feature = repeat_interleave(cat_feature, num_repeats)       # [B*N_ray*N_sample, c0]

        x = torch.cat([x, cat_feature], dim=1)                          # [B*N_ray*N_sample, c0+39]


        # unsqueeze latent_feature, original (SB * NS * B, latent_size)
        now_sampler_points = x.shape[0] // latent_feature.shape[0]                  # sampler_points
        latent_feature = repeat_interleave(latent_feature, now_sampler_points)      # (SB * NS * B * sampler_points, latent_size)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:

                x = x + skip_feature

            if l == 0:
                x = torch.cat([x, latent_feature], dim=1)

                skip_feature = x

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x, latent_feature, cat_feature):
        x.requires_grad_(True)
        y = self.forward(x, latent_feature, cat_feature)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x, latent_feature, cat_feature):
        x.requires_grad_(True)                      ###### get gradients
        output = self.forward(x, latent_feature, cat_feature)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x, latent_feature, cat_feature):
        sdf = self.forward(x, latent_feature, cat_feature)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf