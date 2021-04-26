### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 01f2495e-a379-11eb-2a80-39c0fd68fc96
begin
	using Pkg
	Pkg.activate(".")
end	

# ╔═╡ c4bd0541-889e-482a-a612-184cc468088d
using CairoMakie, DataFrames, MixedModels, PlutoUI

# ╔═╡ df4a9698-1805-438e-9da5-63532874fc9a
md"""
# Bootstrap sampling of LMMs

In this notebook a bootstrap sample from a linear mixed-effects model is used to obtain density plots of the parameter estimates.

Begin by activating the local project
"""

# ╔═╡ 878efa9d-c120-47b8-ab76-72dfe0df4ad7
md"and importing the packages to be used"

# ╔═╡ 280d0d61-ed02-40d7-a6f7-18d3e9f061f8
md"""
The `sleepstudy` dataset is one of those available in the `MixedModels` package.
"""

# ╔═╡ c7e9d58f-1086-4dd7-89ee-64cc31bd3510
slp = MixedModels.dataset(:sleepstudy)

# ╔═╡ f3694c54-596b-458e-b381-2421e20064e1
md"""
The formula for the model provides slope (w.r.t. `days` of sleep deprivation) and intercept as fixed-effects parameters and (possibly correlated) random effects for each of these by `subj`. 
"""

# ╔═╡ dc2e1019-a498-4589-bbff-4b1714962a1b
m1form = @formula(reaction ~ 1 + days + (1+days|subj));

# ╔═╡ f7166b70-11aa-4e55-9e14-68836c63b5ad
m1 = fit(MixedModel, m1form, slp)

# ╔═╡ 668db41d-4def-448d-8c6d-faee71d662a6
md"""
The estimate of the population intercept, which is a typical reaction time for those on their regular sleep schedule, is approximately 250 ms.
The typical reaction time increases by approximately 10.5 ms. per day of sleep deprivation.
The by-subject standard deviation in the intercept is about 20 ms. and that of the slope is about 5.7 ms./day.

The estimate of the correlation, 0.08, is available in the `VarCorr` display
"""

# ╔═╡ 4838527b-d6bd-4548-8666-fc882fb66c1a
VarCorr(m1)

# ╔═╡ e752e80b-f18b-4625-b269-15f34474c8cd
md"""
## Creating a bootstrap sample

The bootstrap sample is a collection of parameter estimates from data sets simulated from this model, with the parameter estimates given above as the "true" parameter values.  For models like this sample sizes of 1000 or more can be used on most laptop computers.
"""

# ╔═╡ 72e0dc34-e907-4658-8f7d-5c2260872455
md""" Size of bootstrap sample: $(@bind sampsz NumberField(100:100:10_000, default=1000))  $(@bind resample PlutoUI.Button("Resample"))
"""

# ╔═╡ 1a831d36-e979-4368-ab1c-e13d94ff1d07
md"Changing the sample size or clicking the `Resample` button will cause a new sample to be generated and the density plots to be refreshed."

# ╔═╡ b56af094-17e9-459f-915f-295d36dceffd
begin
	resample   # causes the block to be re-executed when button is pushed
	samp = parametricbootstrap(sampsz, m1)
	ap = DataFrame(samp.allpars)
	DataFrame(shortestcovint(samp))  # 95% coverage intervals from the samples
end

# ╔═╡ 7fc05850-2ddc-4ebb-954a-7e2d6d3aeadf
md"""
### Density plots

#### Fixed-effects parameters
"""

# ╔═╡ a6b61ca4-dcc0-4e3d-a8cb-9b2775d8b7b5
begin
	fβ = Figure(resolution = (1000, 500))
	βs = filter(:type => ==("β"), ap)
	βnames = unique(βs.names)
	axsβ = [Axis(fβ[1,i]) for i in eachindex(βnames)]
	for (i, k) in enumerate(βnames)
		density!(axsβ[i], filter(:names => ==(k),βs).value)
		axsβ[i].xlabel = k
	end
	fβ
end		

# ╔═╡ d9156571-08c3-4855-9d6c-2331bd829f56
md" #### Standard deviations of the random effects"

# ╔═╡ 9b7b7eb1-02af-446d-8c01-c931ee1b05e6
begin
	fσs = Figure(resolution = (1000, 500))
	dfσ = filter([:type, :group] => (t,g) -> (t == "σ" && g ≠ "residual"), ap)
	σnames = unique(dfσ.names)	
	axsσs = [Axis(fσs[1,i]) for i in eachindex(σnames)]
	for (i, k) in enumerate(σnames)
		density!(axsσs[i], filter(:names => ==(k),dfσ).value)
		axsσs[i].xlabel = string(k)
	end
	fσs
end		

# ╔═╡ c5c7c3b0-4809-41d7-b94d-2bedd1febbcc
md" #### Residual standard deviation and correlation"

# ╔═╡ 824d488e-ad43-4473-8f51-7a6a9863d3b3
begin
	fσ = Figure(resolution = (1000, 500))
	axsσ = Axis(fσ[1,1])
	density!(axsσ, samp.σ)
	axsσ.xlabel = "Residual standard deviation"
	axsρ = Axis(fσ[1,2])
	ρvals = filter(isfinite, filter(:type => ==("ρ"), ap).value)
	hist!(axsρ, ρvals, bins=50, normalization=:density)
	axsρ.xlabel = "Within-subject correlation"
	fσ
end

# ╔═╡ 39fe02d6-f461-4075-aa44-c684f870d635
md"""
The within-subject correlation estimates are shown as a histogram rather than a density plot because of the spike at 1.
"""

# ╔═╡ Cell order:
# ╟─df4a9698-1805-438e-9da5-63532874fc9a
# ╠═01f2495e-a379-11eb-2a80-39c0fd68fc96
# ╟─878efa9d-c120-47b8-ab76-72dfe0df4ad7
# ╠═c4bd0541-889e-482a-a612-184cc468088d
# ╟─280d0d61-ed02-40d7-a6f7-18d3e9f061f8
# ╠═c7e9d58f-1086-4dd7-89ee-64cc31bd3510
# ╟─f3694c54-596b-458e-b381-2421e20064e1
# ╠═dc2e1019-a498-4589-bbff-4b1714962a1b
# ╠═f7166b70-11aa-4e55-9e14-68836c63b5ad
# ╟─668db41d-4def-448d-8c6d-faee71d662a6
# ╠═4838527b-d6bd-4548-8666-fc882fb66c1a
# ╟─e752e80b-f18b-4625-b269-15f34474c8cd
# ╟─72e0dc34-e907-4658-8f7d-5c2260872455
# ╟─1a831d36-e979-4368-ab1c-e13d94ff1d07
# ╠═b56af094-17e9-459f-915f-295d36dceffd
# ╟─7fc05850-2ddc-4ebb-954a-7e2d6d3aeadf
# ╟─a6b61ca4-dcc0-4e3d-a8cb-9b2775d8b7b5
# ╟─d9156571-08c3-4855-9d6c-2331bd829f56
# ╟─9b7b7eb1-02af-446d-8c01-c931ee1b05e6
# ╟─c5c7c3b0-4809-41d7-b94d-2bedd1febbcc
# ╟─824d488e-ad43-4473-8f51-7a6a9863d3b3
# ╟─39fe02d6-f461-4075-aa44-c684f870d635
