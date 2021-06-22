### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 3a1925b2-d361-11eb-281b-5184d2aa9828
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 059344a3-8cb8-4190-b174-689756349150
begin
	using CairoMakie       # 2D graphics from Makie
	using MixedModels      # Fit mixed-effects models; also provides data
	using MixedModelsMakie # Visualization of mixed-model fits
end

# ╔═╡ ae94fbd7-1fbc-4657-9613-aac7a8a4b987
md"""
# Shrinkage of random effects

This notebook illustrates the use of the `MixedModelsMakie` package, especially its `shrinkageplot` and `caterpillarplot` functions, to examine the conditional means of the random effects in large, possibly overparameterized, linear mixed-effects models.

First, load the packages to be used.
"""

# ╔═╡ b0973544-4446-4939-abaf-ba78785ec116
md"""
and the data for the example.

This data set is from experiment 1 reported in Masson, Rabe and Kliegl (2017) *Memory and Cognition*.
"""

# ╔═╡ 1a23a059-2ec3-4108-956b-b1bc1ad8ce7b
MixedModels.dataset(:mrk17_exp1)

# ╔═╡ 7afbbdf4-b3a3-4b9d-870d-9c20d26aaa08
md"""
The experimental or observational units, `subj` and `item`, will be grouping factors for the random effects.
They are assigned `Grouping` contrasts, to bypass the creation of a potentially large contrast matrix that will not be used.
The experimental factors are all two-level factors for which we use the `-1/+1` encoding provided by `EffectsCoding`.
"""

# ╔═╡ f3a111a8-a4c5-467a-aa31-9a86fe1f3b4a
contr = Dict(
    :subj => Grouping(),
    :item => Grouping(),
    :F => EffectsCoding(),
    :P => EffectsCoding(),
    :Q => EffectsCoding(),
    :lQ => EffectsCoding(),
    :lT => EffectsCoding(),
);

# ╔═╡ 7a1704ba-78c6-47ea-b387-7e05eaeae419
md"""
The experimental factors are
- `F`; frequency of the `item` with levels `LF` (low frequency) and `HF` (high frequency)
- `P`: priming with levels `rel` (related) and `unr` (unrelated)
- `Q`: image quality with levels `clr` (clear) and `deg` (degraded)
- `lQ`: lagged quality with the same levels as `Q`
- `lT`: lagged target - was the previous target a word (`WD`) or non-word (`NW`)?

The response time, `rt` [ms] is converted to a speed or rate [responses/s] for analysis.
"""

# ╔═╡ 3cd604e2-be62-40f2-8e6a-fc82812024ba
m1 = fit(
    MixedModel,
    @formula(
        1000 / rt ~
            1 +
            F * P * Q * lQ * lT +
            (1 + P + Q + lQ + lT | item) +
            (1 + F + P + Q + lQ + lT | subj)
    ),
    MixedModels.dataset(:mrk17_exp1);
    contrasts=contr,
)

# ╔═╡ 1fdd2eba-5648-4dad-a441-812bcbff17fb
md"""
## Assessing the parameter estimates

This is a relatively complicated model that can take a few minutes to fit.  Scanning the p-values for the fixed-effects parameters indicates that the fourth and fifth-order interactions can probably be dropped as can all the third-order interactions except for `Q & lQ & lT`.  It is more difficult to assess the significance of the terms in the random effects structure.  The dominant source of variability is `subj` followed by `item` and the `lT | subj` interaction.  (One of the advantages of using a `-1/+1` encoding for the 2-level experimental factors is that the standard deviations of the random effects can be compared directly.)

Even though the correlations of the vector-valued random effects do not appear alarming
"""

# ╔═╡ 1c873918-84c0-4312-ad81-2237626da490
VarCorr(m1)

# ╔═╡ 2d5a2225-89ad-4a20-8a9f-c99b40656b2b
md"this is a singular fit"

# ╔═╡ 1439a0ac-f39a-482b-8e54-16829b163696
issingular(m1)

# ╔═╡ 38d31fdb-7a13-45ef-ac3a-de054c8433b8
md"""
In fact, both of the covariance matrices for the random effects are singular.
One way to assess this is through a principal components analysis.
"""

# ╔═╡ 0a972be6-6186-40a8-9962-1f98b5182f26
m1.PCA

# ╔═╡ c3ad277d-98ca-49fa-b864-bc7510aa9317
md"""
For both `item` and `subj`, the normalized cumulative variance reaches 1.0 before the last component.
Thus, in each case, there is a linear combination of the random effects that is essentially constant.
The linear combination is given by the "loadings" of the last component.
"""

# ╔═╡ 6961baea-44f0-4e9d-b9b3-131e15a8c783
md"""
## Creating the shrinkage plot

The `shrinkageplot` function creates a comparison plot of the conditional means of the random effects at the parameter estimates and the means at a reference parameter value `θref`, chosen so that the covariance, Σ, of the random effects is very large.
The likelihood trades off fidelity to the data, as measured by the sum of squared residuals, versus complexity of the model.
By setting Σ to be very large we are removing the complexity from this balance and and getting nearly unconstrained random effects.
(They are "nearly unconstrained" because there must be a very small amount of shrinkage for identifyability.)
"""

# ╔═╡ 037574f4-f928-4e7a-8e1b-ae8c7fe73335
shrinkageplot(m1)  # vector-valued random effects for item

# ╔═╡ e665dc00-6ff5-4079-bedb-391188fc4f6c
md"""
These patterns for the `item` random effects show that there is moderate shrinkage in the random effects for the `(Intercept)`, in the first column of panels.
In that column, most of the arrows are close to vertical.

The "implosion" patterns on the panels in the other columns show that the random effects are considerably shrunk toward zero.
Thus, these random effects do not have much explanatory power because the considerable shrinkage they experience does not substantially affect the quality of the fit.
"""

# ╔═╡ 8f3be30f-213a-44cd-821d-e720a2ecc083
shrinkageplot(m1, :subj) # vector-valued random effects for subject

# ╔═╡ 57d6f695-c2c8-49d5-a5a4-838ac5933143
md"""
Here, the main conclusion is that the panel on the lower left shows moderate shrinkage in the intercept with respect to `subj` and the `(lT|subj)` interaction.
"""

# ╔═╡ Cell order:
# ╠═3a1925b2-d361-11eb-281b-5184d2aa9828
# ╟─ae94fbd7-1fbc-4657-9613-aac7a8a4b987
# ╠═059344a3-8cb8-4190-b174-689756349150
# ╟─b0973544-4446-4939-abaf-ba78785ec116
# ╠═1a23a059-2ec3-4108-956b-b1bc1ad8ce7b
# ╟─7afbbdf4-b3a3-4b9d-870d-9c20d26aaa08
# ╠═f3a111a8-a4c5-467a-aa31-9a86fe1f3b4a
# ╟─7a1704ba-78c6-47ea-b387-7e05eaeae419
# ╠═3cd604e2-be62-40f2-8e6a-fc82812024ba
# ╟─1fdd2eba-5648-4dad-a441-812bcbff17fb
# ╠═1c873918-84c0-4312-ad81-2237626da490
# ╟─2d5a2225-89ad-4a20-8a9f-c99b40656b2b
# ╠═1439a0ac-f39a-482b-8e54-16829b163696
# ╟─38d31fdb-7a13-45ef-ac3a-de054c8433b8
# ╠═0a972be6-6186-40a8-9962-1f98b5182f26
# ╟─c3ad277d-98ca-49fa-b864-bc7510aa9317
# ╟─6961baea-44f0-4e9d-b9b3-131e15a8c783
# ╠═037574f4-f928-4e7a-8e1b-ae8c7fe73335
# ╟─e665dc00-6ff5-4079-bedb-391188fc4f6c
# ╠═8f3be30f-213a-44cd-821d-e720a2ecc083
# ╟─57d6f695-c2c8-49d5-a5a4-838ac5933143
