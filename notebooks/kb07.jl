### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ e1f9e6d8-8993-11eb-01fd-11021715dfc7
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 154380a8-899e-11eb-2601-a5b15d7c351e
using AlgebraOfGraphics, CairoMakie, DataFrames, MixedModels, Random

# ╔═╡ a1076b78-8993-11eb-176b-63b0349cff6d
md"""
# Bootstrapping a fitted model

Begin by activating the package environment for this project and the packages to be used.
"""

# ╔═╡ b123fc8c-8994-11eb-0b47-b572d73b9fba
md"Provide a short alias for `AlgebraOfGraphics` to make expressions easier"

# ╔═╡ 05721daa-8995-11eb-2b3e-7d7fb0dd6beb
const AOG = AlgebraOfGraphics;

# ╔═╡ 0f965792-8995-11eb-0e9d-71a7600e3771
md"""
## Data set and model

The `kb07` data are one of the datasets provided as *columntable*s by the `MixedModels` package.
Here the table is converted to a `DataFrame` for display.
"""

# ╔═╡ 3c7744ec-8995-11eb-0b3f-29e094fe0293
kb07 = DataFrame(MixedModels.dataset(:kb07))

# ╔═╡ f7dac810-89a1-11eb-384f-2b3855d24a8c
describe(kb07)

# ╔═╡ 3be06f70-8997-11eb-1f76-19b9c65ebb16
md"""
The experimental factors; `spkr`, `prec`, and `load`, are two-level factors.  The `EffectsCoding` contrast is used with these to create a $\pm1$ encoding.
Furthermore, the `subj` and `item` factors are specified as grouping factors.
This is done to avoid memory overflow when using grouping factors with a very large number of levels when creating a contrast matrix that is not going to be used.
It is not important in these cases but a good practice in any case.
"""

# ╔═╡ 0c0ede16-8998-11eb-2b9b-3bfde8213d42
kbcontr = merge(
	Dict(nm => EffectsCoding() for nm in (:spkr, :prec, :load)),
	Dict(nm => Grouping() for nm in (:subj, :item)),
	);

# ╔═╡ 49597c2c-8998-11eb-3be6-a30ceeb67d89
md"""
An initial model formula of
"""

# ╔═╡ bf6bdb58-8998-11eb-39d9-cb774d3f02af
kb1form = @formula(
	rt_trunc ~ 1+spkr*prec*load + (1+spkr+prec+load|subj) + (1+spkr+prec+load|item)
	);

# ╔═╡ 0f3cd8b0-8999-11eb-0a21-f9f976de4db9
md"produces the fitted model"

# ╔═╡ 2217730c-8999-11eb-35a4-01d6e1efe14b
kbm1 = fit(MixedModel, kb1form, kb07, contrasts=kbcontr)

# ╔═╡ f91301ba-899f-11eb-0586-f1163fee3d4d
md"""
The estimated correlations of the random effects are not displayed in this table.
The `VarCorr` extractor displays these.
"""

# ╔═╡ c26a003a-899f-11eb-39cc-19a4e3964fe9
VarCorr(kbm1)

# ╔═╡ d0c5090a-899b-11eb-123d-97101e374752
md"""
None of the two-factor or three-factor interaction terms in the fixed-effects are significant.
In the random-effects terms only the scalar random effects and the `prec` random effect for `item` appear to be warranted, leading to the reduced formula
"""

# ╔═╡ a15ab608-899c-11eb-3962-1984b285c11d
kb2form = @formula(rt_trunc ~ 1+spkr+prec+load+(1|subj)+(1+prec|item));

# ╔═╡ cc7ef8d0-899c-11eb-3288-df5fcfd5ca49
kbm2 = fit(MixedModel, kb2form, kb07, contrasts=kbcontr)

# ╔═╡ d2e2be98-899f-11eb-13b2-9daf138106d1
VarCorr(kbm2)

# ╔═╡ 4de8f136-89a0-11eb-303e-cd2ebd36c6cd
md"The two models are nested and can be compared with a likelihood-ratio test."

# ╔═╡ 62f4f320-89a0-11eb-15ae-6faa3eb0724c
MixedModels.likelihoodratiotest(kbm2, kbm1)

# ╔═╡ 7b3a8080-89a0-11eb-371d-75212efaf3c7
md"""
The p-value of approximately 17% leads us to prefer the simpler model, `kbm2`, to the more complex, `kbm1`.
"""

# ╔═╡ 932de472-899e-11eb-3930-efba479748fa
md"""
## A bootstrap sample

Create a bootstrap sample of a few thousand parameter estimates from the reduced model.
The `MersenneTwister` pseudo-random number generator is initialized to a fixed value for reproducibility.
"""

# ╔═╡ 299d4358-899f-11eb-2b8c-4d6c3a25b101
kb2bstrp = parametricbootstrap(MersenneTwister(1234321), 2000, kbm2); 

# ╔═╡ 2f6b4306-89a0-11eb-3ba9-133035ac2d33
md"""
One of the uses of such a sample is to form "confidence intervals" on the parameters by obtaining the shortest interval that covers a given proportion (95%, by default) of the sample.
"""

# ╔═╡ 630b49c8-899f-11eb-08fd-0b65c3b2127e
DataFrame(shortestcovint(kb2bstrp))

# ╔═╡ 5ee005ee-89a1-11eb-0690-577497a0edaa
md"""
A sample like this can be used for more than just creating an interval because it approximates the distribution of the estimator.
For the fixed-effects parameters the estimators are close to being normally distributed.
"""

# ╔═╡ a4e04248-89a1-11eb-27da-f390c2544ca9
data(kb2bstrp.β) * mapping(:β; layout_x = :coefname) * AOG.density |> draw

# ╔═╡ 4cbcdb5c-89a2-11eb-3522-c3645690afe2
data(filter(:column => ==(Symbol("(Intercept)")), DataFrame(kb2bstrp.σs))) * mapping(:σ, layout_x = :group) * AOG.density |> draw

# ╔═╡ e7afbe94-89a6-11eb-1194-75349e639c70
DataFrame(kb2bstrp.σs)

# ╔═╡ c0b321e2-89a2-11eb-3af4-273b108abbf0
data(filter(:type => ==("ρ"), DataFrame(kb2bstrp.allpars))) * mapping(:value, layout_x = :names) * AOG.density |> draw

# ╔═╡ Cell order:
# ╟─a1076b78-8993-11eb-176b-63b0349cff6d
# ╠═e1f9e6d8-8993-11eb-01fd-11021715dfc7
# ╠═154380a8-899e-11eb-2601-a5b15d7c351e
# ╟─b123fc8c-8994-11eb-0b47-b572d73b9fba
# ╠═05721daa-8995-11eb-2b3e-7d7fb0dd6beb
# ╟─0f965792-8995-11eb-0e9d-71a7600e3771
# ╠═3c7744ec-8995-11eb-0b3f-29e094fe0293
# ╠═f7dac810-89a1-11eb-384f-2b3855d24a8c
# ╟─3be06f70-8997-11eb-1f76-19b9c65ebb16
# ╠═0c0ede16-8998-11eb-2b9b-3bfde8213d42
# ╟─49597c2c-8998-11eb-3be6-a30ceeb67d89
# ╠═bf6bdb58-8998-11eb-39d9-cb774d3f02af
# ╟─0f3cd8b0-8999-11eb-0a21-f9f976de4db9
# ╠═2217730c-8999-11eb-35a4-01d6e1efe14b
# ╟─f91301ba-899f-11eb-0586-f1163fee3d4d
# ╠═c26a003a-899f-11eb-39cc-19a4e3964fe9
# ╟─d0c5090a-899b-11eb-123d-97101e374752
# ╠═a15ab608-899c-11eb-3962-1984b285c11d
# ╠═cc7ef8d0-899c-11eb-3288-df5fcfd5ca49
# ╠═d2e2be98-899f-11eb-13b2-9daf138106d1
# ╟─4de8f136-89a0-11eb-303e-cd2ebd36c6cd
# ╠═62f4f320-89a0-11eb-15ae-6faa3eb0724c
# ╟─7b3a8080-89a0-11eb-371d-75212efaf3c7
# ╟─932de472-899e-11eb-3930-efba479748fa
# ╠═299d4358-899f-11eb-2b8c-4d6c3a25b101
# ╟─2f6b4306-89a0-11eb-3ba9-133035ac2d33
# ╠═630b49c8-899f-11eb-08fd-0b65c3b2127e
# ╟─5ee005ee-89a1-11eb-0690-577497a0edaa
# ╠═a4e04248-89a1-11eb-27da-f390c2544ca9
# ╠═4cbcdb5c-89a2-11eb-3522-c3645690afe2
# ╠═e7afbe94-89a6-11eb-1194-75349e639c70
# ╠═c0b321e2-89a2-11eb-3af4-273b108abbf0
