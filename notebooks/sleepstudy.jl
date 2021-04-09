### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ f9030e18-e484-11ea-24e0-3132afabdf83
begin
	using Pkg
	Pkg.activate(".")
	using MixedModels
end		

# ╔═╡ 1357b670-e484-11ea-2275-cba956358da4
md"# Analysis of the sleepstudy data"

# ╔═╡ 2a6a0106-e484-11ea-2d1c-65ea0604a1e3
md"""
The `sleepstudy` data are from a study on the effects of sleep deprivation on
response time.  Eighteen subjects were allowed only 3 hours of time to sleep each night for 9 successive nights.  Their reaction time was measured each day, starting the day before the first night of sleep deprivation, when the subjects were on their regular sleep schedule.
"""

# ╔═╡ bf6eab08-e484-11ea-21b3-79b2acabd22f
md"## Loading the data"

# ╔═╡ 44d5f548-e4ae-11ea-13c2-ddb1bbed274e
md"""
First attach the MixedModels package.  In Pluto this must be done within a `begin ... end` block in a cell.
"""

# ╔═╡ 786e3abe-e4ae-11ea-0965-6bbd681d0f72
md"""
The `sleepstudy` data are one of the datasets available with recent versions of the `MixedModels` package.
"""

# ╔═╡ 66a6f3bc-e485-11ea-2b47-13a48b5d7e52
sleepstudy = MixedModels.dataset("sleepstudy")

# ╔═╡ f3ed171c-e4ab-11ea-1eac-8b0265a6c6ea
md"## Fitting an initial model"

# ╔═╡ 8a902898-e485-11ea-2d50-b3c44d666f9c
m1 = fit(MixedModel, @formula(reaction ~ 1+days+(1+days|subj)), sleepstudy)

# ╔═╡ 06560c54-e4ac-11ea-3b36-e57ee4bc3134
md"""
This model includes fixed effects for the intercept, representing the typical reaction time at the beginning of the experiment with zero days of sleep deprivation, and the slope w.r.t. days of sleep deprivation.  In this case about 250 ms. typical reaction time without deprivation and a typical increase of 10.5 ms. per day of sleep deprivation.

The random effects represent shifts from the typical behavior for each subject.The shift in the intercept has a standard deviation of about 24 ms. which would indicate a range of about 200 ms. to 300 ms. in the intercepts.  Similarly within-subject slopes would be expected to have a range of about 0 ms./day up to 20 ms./day.

The within-subject correlation of the random effects for intercept and slope is small, +0.8, indicating that a simpler model with uncorrelated random effects may be sufficient.
"""

# ╔═╡ c01a9062-e4ac-11ea-383c-471462a1e649
md"## A model with uncorrelated random effects"

# ╔═╡ 3e3b227c-e4ad-11ea-3c24-fb23d26e1171
md"""
The `zerocorr` function applied to a random-effects term creates uncorrelated vector-valued per-subject random effects.
"""

# ╔═╡ d68c3c14-e485-11ea-0b41-6d234ab4b676
m2 = fit(MixedModel, @formula(reaction ~ 1+days+zerocorr(1+days|subj)),sleepstudy)

# ╔═╡ 8e066b36-e4ad-11ea-2655-47a9a8d3c031
md"""
This model has a slightly lower log-likelihood than does `m1` but one less parameter than `m1`.   A likelihood-ratio test can be used to compare these nested models.
"""

# ╔═╡ ec436546-e485-11ea-19b8-a15f2f8247df
MixedModels.likelihoodratiotest(m2, m1)

# ╔═╡ d0856ae0-e4ab-11ea-0016-2f170c65768f
md"""
Alternatively, the AIC or BIC values can be compared.  They are on a scale where "smaller is better".  Both these model-fit statistics prefer `m2`, the simpler model.
"""

# ╔═╡ Cell order:
# ╟─1357b670-e484-11ea-2275-cba956358da4
# ╟─2a6a0106-e484-11ea-2d1c-65ea0604a1e3
# ╟─bf6eab08-e484-11ea-21b3-79b2acabd22f
# ╟─44d5f548-e4ae-11ea-13c2-ddb1bbed274e
# ╠═f9030e18-e484-11ea-24e0-3132afabdf83
# ╟─786e3abe-e4ae-11ea-0965-6bbd681d0f72
# ╠═66a6f3bc-e485-11ea-2b47-13a48b5d7e52
# ╟─f3ed171c-e4ab-11ea-1eac-8b0265a6c6ea
# ╠═8a902898-e485-11ea-2d50-b3c44d666f9c
# ╟─06560c54-e4ac-11ea-3b36-e57ee4bc3134
# ╟─c01a9062-e4ac-11ea-383c-471462a1e649
# ╟─3e3b227c-e4ad-11ea-3c24-fb23d26e1171
# ╠═d68c3c14-e485-11ea-0b41-6d234ab4b676
# ╟─8e066b36-e4ad-11ea-2655-47a9a8d3c031
# ╠═ec436546-e485-11ea-19b8-a15f2f8247df
# ╟─d0856ae0-e4ab-11ea-0016-2f170c65768f
