### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ f9030e18-e484-11ea-24e0-3132afabdf83
begin
	using Arrow, DataFrames, DataFramesMeta, CategoricalArrays 
	using Statistics, LinearAlgebra
	using MixedModels, MixedModelsMakie
	using AlgebraOfGraphics, CairoMakie 
end		

# ╔═╡ 1357b670-e484-11ea-2275-cba956358da4
md"# The Sleepstudy: Speed - for a change ..."

# ╔═╡ 2a6a0106-e484-11ea-2d1c-65ea0604a1e3
md"""
## Background

Belenky et al. (2003) reported effects of sleep deprivation across a 14-day study of 30-to-40-year old men and women holding commercial vehicle driving licenses. Their analyses are based on a subset of tasks and ratings from very large and comprehensive test and questionnaire battery (Balkin et al., 2000).

Initially 66 subjects were assigned to one of four time-in-bed (TIB) groups with 9 hours  (22:00-07:00) of sleep augmentation or 7 hours (24:00-07:00), 5 hours (02:00-07:00), and 3 hours (04:00-0:00) of sleep restrictions per night, respectively. The final sample comprised 56 subjects. The Psychomotor Vigilance Test (PVT) measures simple reaction time to a visual stimulus, presented approximately 10 times ⁄ minute (interstimulus interval varied from 2 to 10 s in 2-s increments) for 10 min and implemented in a thumb-operated, hand-held device (Dinges and Powell 1985).

**Design**

The study comprised 2 training days (T1, T2), one day with baseline measures (B), seven days with sleep deprivation (E1 to E7), and four recovery days (R1 to R4). T1 and T2 were devoted to training on the performance tests and familiarization with study procedures. PVT baseline testing commenced on the morning of the third day (B) and testing continued for the duration of the study (E1–E7, R1–R3; no measures were taken on R4). Bed times during T, B, and R days were 8 hours (23:00-07:00).

**Test schedule within days**

The PVT (along with the Stanford Sleepiness Scale) was administered as a battery four times per day (09:00, 12:00, 15:00, and 21:00 h); the battery included other tests not reported here (see Balkin et al. 2000). The sleep latency test was administered at 09:40 and 15:30 h for all groups. Subjects in the 3- and 5-h TIB groups performed an additional battery at 00:00 h and 02:00 h to occupy their additional time awake. The PVT and SSS were administered in this battery; however, as data from the 00:00 and 02:00 h sessions were not common to all TIB groups, these data were not included in the statistical analyses reported in the paper.

**Statistical analyses**

The authors analyzed response speed, that is (1/RT)*1000 -- completely warranted according to a Box-Cox check of the current data -- with mixed-model ANOVAs using group as between- and day as within-subject factors. The ANOVA was followed up with simple tests of the design effects implemented over days for each of the four groups.

**Current data**

The current data distributed with the _RData_ collection is attributed to the 3-hour TIB group, but the means do not agree at all with those reported for this group in Belenky et al. (2003, Figure 3) where the 3-hour TIB group is also based on only 13 (not 18) subjects. Specifically, the current data show a much smaller slow-down of response speed across E1 to E7 and do not reflect the recovery during R1 to R3. The currrent data also cover only 10 not 11 days, but it looks like only R3 is missing. The closest match of the current means was with the average of the 3-hour and 7-hour TIB groups; if only males were included, this would amount to 18 subjects. (This conjecture is based only on visual inspection of graphs.) 

**References**

+ Balkin, T., et al. (2000). _Effects of sleep schedules on commercial motor vehicle driver performance._  Report MC-00–133, National Technical Information Service, U.S. Department of Transportation, Springfield, VA.

+ Belenky, G., et al. (2003). Patterns of performance degradation and restoration during sleep restriction and subsequent recovery: a sleep dose-response study. _Journal of Sleep Research_, **12**, 1-12.

+  Dinges, D. F. and Powell, J. W. (1985). Microcomputer analyses of performance on a portable, simple, visual RT task during sustained operations. _Behavior Research Methods, Instrumentation, and Computers_, **17**, 652–655.

"""

# ╔═╡ bf6eab08-e484-11ea-21b3-79b2acabd22f
md"## Setup"

# ╔═╡ 44d5f548-e4ae-11ea-13c2-ddb1bbed274e
md"""
First we attach the various packages needed, define a few helper functions, read the data, and get everything in the desired shape.
"""

# ╔═╡ ae074801-0697-4811-baef-76e8ca8fac86
function viewdf(x)
		DataFrame(Arrow.Table(take!(Arrow.write(IOBuffer(), x))))
end

# ╔═╡ 5d02bc2b-5ae4-4017-a482-a35fb144b3cf
md" Two simple regression functions. **RK:** Why does the first one not work here?"

# ╔═╡ 51b19e16-b24c-4332-b5b8-6d5a5605f397
linreg1(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat} = [ones(length(x)) x]\y

# ╔═╡ f1d7982a-0a5c-40ae-8e09-9976cb49ea0f
function simplelinreg(x, y)
    A = cholesky!(Symmetric([length(x) sum(x) sum(y); 0.0 sum(abs2, x) dot(x, y); 0.0 0.0 sum(abs2, y)])).factors
    ldiv!(UpperTriangular(view(A, 1:2, 1:2)), view(A, 1:2, 3))
end

# ╔═╡ e09a7f5c-1b17-4d60-958c-b84807c6638e
md"## Preprocessing"

# ╔═╡ 786e3abe-e4ae-11ea-0965-6bbd681d0f72
md"""
The `sleepstudy` data are one of the datasets available with recent versions of the `MixedModels` package. We carry out some preprocessing to have the dataframe in the desired shape:
+ Capitalize random factor `Subj` (-- is not a factor yet --)
+ Compute `speed` as an alternative dependent variable from `reaction`, warranted by a 'boxcox' check of residuals. 
+ Compute a sort index giving for `Subj`s' mean `speed`
+ Covert `Subj` to CategoricalArray (i.e., factor)
+ Relevel `Subj` according to speed-based sort index
"""

# ╔═╡ 66a6f3bc-e485-11ea-2b47-13a48b5d7e52
begin
	data = DataFrame(MixedModels.dataset("sleepstudy"));
	rename!(data, :subj => :Subj, :days => :day);
    transform!(data, :reaction, :reaction => (x -> 1000 ./ x) => :speed);
	ix = sort(combine(groupby(data, :Subj), :speed => mean), :speed_mean);
	data = @linq data |> transform(Subj = levels!(categorical(:Subj), ix.Subj));
	sort!(data, :Subj);
	viewdf(data)
end

# ╔═╡ 61076594-0398-40c9-9190-08c820c6da97
describe(data)

# ╔═╡ 44adec0b-eec0-4b85-9161-01c11525c831
md""" ## Estimates for pooled data

In the first analysis we ignore the dependency of observvations due to repeated measures from the same subjects. We pool all the data and estimate the regression of 180 speed scores on the nine dayse of the experiment. We store the estimates for each `Subj`.
"""

# ╔═╡ da75674a-aca1-4aa1-8d71-5dde799f7e90
begin
	coef = simplelinreg(data.day, data.speed);
	pld = DataFrame(Subj=ix.Subj, day_0=coef[1], effect=coef[2], 
		  estimate = "Pooled");
end

# ╔═╡ 6d60ce3a-2ac9-447c-8a44-d185b8562a2d
md"""
## Within-subject effects

In the second analysis we estimate coefficients for each `Subj` without regard of the information available from the complete set of data. We do not "borrow strength" to adjust for differences due to between-`Subj` variability and due to being far from the population mean. 

### Within-subject simple regressions

+ Define dataframe to hold within-`Subj` effects
+ Compute the simple regression of speed on day for every `Subj` (i.e., the within-subject effects of sleep deprivation). 

The `(Intercept)` returns the estimate of `speed` at day 0, that is an estimate of `speed` prior to the experimental days, assuming that`day_0` corresponds to day B in the original study. 
"""

# ╔═╡ 651afde1-6fe6-418e-bcd4-f95c167d7305
begin
  wss = DataFrame(Subj=ix.Subj, day_0 = .0, effect = .0, estimate = "within-Subj" );
  for i in 1:length(ix.Subj)
  	local subj_df
  	subj_df = filter(row -> row.Subj == ix.Subj[i], data) 
  	wss[i, 2:3] = simplelinreg(subj_df.day, subj_df.speed)
  end
  wss
end

# ╔═╡ 071124e6-ba1c-4d83-b5fc-97e7a45f762c
md" ### Figure of within-subject data and simple regressions"

# ╔═╡ 7e4f18b9-e226-45f7-9d85-45ff9f7a6992
begin
	fSubj = Figure(resolution = (1600,1200))
	x = LinRange(0, 9, 10)
	y = LinRange(2, 5.5, 8) 
	local faxs
	faxs =  [Axis(fSubj[1, i]) for i in 1:6]
	for i in 1:6
		  Box(fSubj[1, i, Top()], backgroundcolor = :gray)
		Label(fSubj[1, i, Top()], wss.Subj[i], padding = (5, 5, 5, 5))
		local subj_df
  	    subj_df = filter(row -> row.Subj == ix.Subj[i], data) 
		scatter!(faxs[i], subj_df.day, subj_df.speed)
		local y
		y = wss.day_0[i] .+ wss.effect[i] .* x
		lines!(faxs[i], x, y,  color = :blue, linewidth = 4)
	end
	faxs[3].xlabel = "Day"                         # only one x-axis label
	faxs[1].ylabel = "Speed [1000/RT]"             # only one y-axis label
	hideydecorations!.(faxs[2:end], grid = false)  # ylab only on left panel
	linkaxes!(faxs...)                             # use the same axes 
	colgap!(fSubj.layout, 10);                      # tighten spacing btw panels
end

# ╔═╡ ede622d9-36ec-4950-90af-d7b6ac0cbd3d
begin
	local faxs
	faxs =  [Axis(fSubj[2, i]) for i in 1:6]
	for i in 1:6
		  Box(fSubj[2, i, Top()], backgroundcolor = :gray)
		Label(fSubj[2, i, Top()], wss.Subj[i+6], padding = (5, 5, 5, 5))
		local subj_df
  	    subj_df = filter(row -> row.Subj == ix.Subj[i+6], data) 
		scatter!(faxs[i], subj_df.day, subj_df.speed)
		local y
		y = wss.day_0[i+6] .+ wss.effect[i+6] .* x
		lines!(faxs[i], x, y,  color = :blue, linewidth = 4)
	end
	faxs[3].xlabel = "Day"                         # only one x-axis label
	faxs[1].ylabel = "Speed [1000/RT]"             # only one y-axis label
	hideydecorations!.(faxs[2:end], grid = false)  # ylab only on left panel
	linkaxes!(faxs...)                             # use the same axes 
	colgap!(fSubj.layout, 10)                       # tighten spacing btw panels
end

# ╔═╡ 2b5e7a1c-4798-4734-af15-b5aa1029e1e7
begin
	local faxs
	faxs =  [Axis(fSubj[3, i]) for i in 1:6]
	for i in 1:6
		  Box(fSubj[3, i, Top()], backgroundcolor = :gray)
		Label(fSubj[3, i, Top()], wss.Subj[i+12], padding = (5, 5, 5, 5))
		local subj_df
  	    subj_df = filter(row -> row.Subj == ix.Subj[i+12], data) 
		scatter!(faxs[i], subj_df.day, subj_df.speed)
		local y
		y = wss.day_0[i+12] .+ wss.effect[i+12] .* x
		lines!(faxs[i], x, y,  color = :blue, linewidth = 4)
	end
	faxs[3].xlabel = "Day"                         # only one x-axis label
	faxs[1].ylabel = "Speed [1000/RT]"             # only one y-axis label
	hideydecorations!.(faxs[2:end], grid = false)  # ylab only on left panel
	linkaxes!(faxs...)                             # use the same axes 
	colgap!(fSubj.layout, 10)                      # tighten spacing btw panels
	fSubj
end

# ╔═╡ f3ed171c-e4ab-11ea-1eac-8b0265a6c6ea
md"## Basic LMM"

# ╔═╡ 8a902898-e485-11ea-2d50-b3c44d666f9c
m1 = fit(MixedModel, @formula(speed ~ 1+day+(1+day|Subj)), data)

# ╔═╡ 06560c54-e4ac-11ea-3b36-e57ee4bc3134
md"""
This model includes fixed effects for the intercept which estimates the average speed on the baseline day of the experiment prior to sleep deprivation, and the slowing per day of sleep deprivation. In this case about -0.11/second. 

The random effects represent shifts from the typical behavior for each subject.The shift in the intercept has a standard deviation of about 0.42/s. 

The within-subject correlation of the random effects for intercept and slope is small, -0.18, indicating that a simpler model with a correlation parameter (CP) forced to/ assumed to be zero may be sufficient.
"""

# ╔═╡ c01a9062-e4ac-11ea-383c-471462a1e649
md"##  No correlation parameter: zcp LMM"

# ╔═╡ 3e3b227c-e4ad-11ea-3c24-fb23d26e1171
md"""
The `zerocorr` function applied to a random-effects term estimates one paremeter less than LMM `m1`-- the CP is now fixed to zero.
"""

# ╔═╡ d68c3c14-e485-11ea-0b41-6d234ab4b676
m2 = fit(MixedModel, @formula(speed ~ 1+day+zerocorr(1+day|Subj)),data)

# ╔═╡ 8e066b36-e4ad-11ea-2655-47a9a8d3c031
md"""
LMM `m2` has a slghtly  lower log-likelihood than LMM `m1` but also one parameter less.  The likelihood-ratio test is used to compare these nested models.
"""

# ╔═╡ ec436546-e485-11ea-19b8-a15f2f8247df
MixedModels.likelihoodratiotest(m2, m1)

# ╔═╡ d0856ae0-e4ab-11ea-0016-2f170c65768f
md"""
Alternatively, the AIC, AICc, and BIC values can be compared.  They are on a scale where "smaller is better".  All three model-fit statistics prefer the zcpLMM `m2`.
"""

# ╔═╡ 9b0956f5-e02f-4235-ba24-49e75fe71169
DataFrame(dof=dof.([m2, m1]), deviance=deviance.([m2, m1]),
          AIC = aic.([m2, m1]), AICc = aicc.([m2, m1]), BIC = bic.([m2, m1]))

# ╔═╡ a17bb040-d483-479d-8bbd-36f723f67a66
md"""
## Conditional modes of the random effects

The third set of estimates are their conditional modes. They represent a compromise between their own data and the model parameters. When distributional assumptions hold, predictions based on these estimates are more accurate than either the pooled or the within-subject estimates. Here we "borrow strength" to improve the accuracy of prediction.

"""

# ╔═╡ 7d74a757-8336-4476-9f2e-4c2493cf2b5e
cms = wss

# ╔═╡ 0fb18a9c-7fda-45bf-bd29-c341a41854d2
md" ## Combine the three types of estimates" 

# ╔═╡ 3abc830c-4e1a-4d09-beec-46e6ff9f6cbf
begin
	cms_wss_pld = vcat(cms, wss, pld)
	cms_wss_pld = @linq cms_wss_pld |>
       transform(
       estimate = levels!(categorical(:estimate), 
		["Conditional mean", "within-Subj", "Pooled"])
       );
	viewdf(cms_wss_pld)
end

# ╔═╡ bc413a49-d082-40aa-8c9b-e3d64904f6cf
md" ## Three types of individual response profiles"

# ╔═╡ dab9ed8c-1deb-495b-901c-59a0c7ffcb26
md"""
Here we compare the three types on individual response profiles -- pooled, within-`Subj`, and based on conditional means of random effects.
"""

# ╔═╡ 87fd5191-9ff9-4f34-a6ed-eda19c7a6ae2
md" ## Caterpillar plots (effect profiles)"

# ╔═╡ 31b8e4c1-55d7-47b6-850a-d5356490b046
begin
	cm_m2 = ranefinfo(m2)[:Subj];
	caterpillar!(Figure(; resolution=(800,600)), cm_m2; orderby=1)
end

# ╔═╡ 168bcc03-7969-4b6a-bb2c-c1d4c18a9f4f
md" ## Shrinkage plot"

# ╔═╡ d29f6377-e8ae-40dc-b084-d31e14728edd
begin
	shrnk_m2 = shrinkage(m2)[:Subj];
	shrinkageplot!(Figure(; resolution=(800,600)), shrnk_m2)
end

# ╔═╡ Cell order:
# ╟─1357b670-e484-11ea-2275-cba956358da4
# ╟─2a6a0106-e484-11ea-2d1c-65ea0604a1e3
# ╟─bf6eab08-e484-11ea-21b3-79b2acabd22f
# ╟─44d5f548-e4ae-11ea-13c2-ddb1bbed274e
# ╠═f9030e18-e484-11ea-24e0-3132afabdf83
# ╠═ae074801-0697-4811-baef-76e8ca8fac86
# ╠═5d02bc2b-5ae4-4017-a482-a35fb144b3cf
# ╠═51b19e16-b24c-4332-b5b8-6d5a5605f397
# ╠═f1d7982a-0a5c-40ae-8e09-9976cb49ea0f
# ╟─e09a7f5c-1b17-4d60-958c-b84807c6638e
# ╟─786e3abe-e4ae-11ea-0965-6bbd681d0f72
# ╟─66a6f3bc-e485-11ea-2b47-13a48b5d7e52
# ╠═61076594-0398-40c9-9190-08c820c6da97
# ╟─44adec0b-eec0-4b85-9161-01c11525c831
# ╟─da75674a-aca1-4aa1-8d71-5dde799f7e90
# ╟─6d60ce3a-2ac9-447c-8a44-d185b8562a2d
# ╟─651afde1-6fe6-418e-bcd4-f95c167d7305
# ╟─071124e6-ba1c-4d83-b5fc-97e7a45f762c
# ╠═7e4f18b9-e226-45f7-9d85-45ff9f7a6992
# ╠═ede622d9-36ec-4950-90af-d7b6ac0cbd3d
# ╠═2b5e7a1c-4798-4734-af15-b5aa1029e1e7
# ╟─f3ed171c-e4ab-11ea-1eac-8b0265a6c6ea
# ╠═8a902898-e485-11ea-2d50-b3c44d666f9c
# ╟─06560c54-e4ac-11ea-3b36-e57ee4bc3134
# ╟─c01a9062-e4ac-11ea-383c-471462a1e649
# ╟─3e3b227c-e4ad-11ea-3c24-fb23d26e1171
# ╠═d68c3c14-e485-11ea-0b41-6d234ab4b676
# ╟─8e066b36-e4ad-11ea-2655-47a9a8d3c031
# ╠═ec436546-e485-11ea-19b8-a15f2f8247df
# ╟─d0856ae0-e4ab-11ea-0016-2f170c65768f
# ╠═9b0956f5-e02f-4235-ba24-49e75fe71169
# ╟─a17bb040-d483-479d-8bbd-36f723f67a66
# ╠═7d74a757-8336-4476-9f2e-4c2493cf2b5e
# ╟─0fb18a9c-7fda-45bf-bd29-c341a41854d2
# ╟─3abc830c-4e1a-4d09-beec-46e6ff9f6cbf
# ╠═bc413a49-d082-40aa-8c9b-e3d64904f6cf
# ╠═dab9ed8c-1deb-495b-901c-59a0c7ffcb26
# ╟─87fd5191-9ff9-4f34-a6ed-eda19c7a6ae2
# ╠═31b8e4c1-55d7-47b6-850a-d5356490b046
# ╟─168bcc03-7969-4b6a-bb2c-c1d4c18a9f4f
# ╠═d29f6377-e8ae-40dc-b084-d31e14728edd
