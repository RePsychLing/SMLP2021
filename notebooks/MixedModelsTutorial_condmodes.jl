### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 9396fcac-b0b6-11eb-3a60-9f2ce25df953
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 4ecf1267-aac9-4e8f-a522-5595b1205f43
begin
	using CSV, RCall, DataFrames, MixedModels
	using LinearAlgebra, Statistics, StatsBase
	using CategoricalArrays, Arrow
	using AlgebraOfGraphics
	using AlgebraOfGraphics: linear
	using CairoMakie  # for Scatter
end

# ╔═╡ eebef932-1d63-41e5-998f-9748379c43af
md"""
# Mixed Models Tutorial: Conditional Modes

Ths script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

To circumvent delays associated with model fitting we work with a reduced data set and less complex models than those in the reference publication. All the data to reproduce the models in the publication are used here, too; the script requires only a few changes to specify the more complex models in the paper. 

The script is structured in three main sections: 

1. **Setup** with reading and examing the data, plotting the main results, and specifying the contrasts for the fixed factor `Test`.
2. **Extraction of conditional modes** 
3. **Caterpillar plot**

## 1. Setup
### 1.0 Packages and functions
"""

# ╔═╡ faafd359-fb38-4f49-9dfd-19cb4f0c5a54
begin
	function viewdf(x)
		DataFrame(Arrow.Table(take!(Arrow.write(IOBuffer(), x))))
	end
end

# ╔═╡ ee85abd9-0172-44d3-b03a-c6a780d33c72
md"""
### 1.1 Readme for 'EmotikonSubset.rds'

1. Cohort: 9 levels; 2011-2019
2. School: 46 levels 
3. Child: 11566 levels; all children were between 6.0 and 6.99 years at legal keydate (30 September) of school enrollement.
4. Sex: 5893 girls, 5673 boys
5. age: test date - middle of month of birthdate (ranges between 7.9 and 9.2)
6. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 7. score - see units

### 1.2 Preprocessing
#### Read data
"""

# ╔═╡ 39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
dat = rcopy(R"readRDS('./data/EmotikonSubset.rds')");

# ╔═╡ 0051ec83-7c30-4d28-9dfa-4c6f5d0259fa
md"""
#### Transformations
"""

# ╔═╡ 7440c439-9d4c-494f-9ac6-18c9fb2fe144
begin
	transform!(dat, :age, :age => (x -> x .- 8.5) => :a1); # centered age (linear)
	transform!(dat,  :a1, :a1  => (x -> x.^2) => :a2);     # centered age (quadr.)
	select!(groupby(dat,  :Test), :, :score => zscore => :zScore); # z-score
end;

# ╔═╡ 16c3cdaa-49fa-46d5-a844-03094329fe4c
viewdf(dat)

# ╔═╡ 64cc1f8e-f831-4a53-976f-dc7600b5634d
# ... by Test and Sex
begin
	dat2 = combine(groupby(dat, [:Test, :Sex]), 
                             :score => mean, :score  => std, 
                             :zScore => mean, :zScore => std)
	viewdf(dat2)
end

# ╔═╡ c6c6f056-b8b9-4190-ac14-b900bafa04df
md"""
## 2 LMMs and extraction of conditional modes 
### 2.1 _SeqDiffCoding_ of `Test`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise 
differences between the five neighboring levels of `Test`, that is: 

+ H1: `Star_r` - `Run` (2-1)
+ H2: `S20_r` - `Star_r` (3-2) 
+ H3: `SLJ` - `S20_r` (4-3)
+ H4: `BPT` - `SLJ` (5-4)

Various options for contrast coding are the topic of the *MixedModelsTutorial_contrasts.jl*
notbebook.
"""

# ╔═╡ c5326753-a03b-4739-a82e-90ffa7c1ebdb
begin
	contr = merge(
        Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	    Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"])),
        Dict(:TestHC => HelmertCoding(; levels=["S20_r", "SLJ", "Star_r", "Run", "BPT"])),
	   )
end;

# ╔═╡ e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
md"""
#### 2.2 LMMs  `m11` for the reduced data

We fit LMM `m11_Cohort` (only nine levels, six VCs for Test; no CPs) and `m11_School` (46 levels; VCs and CPs for test and linear effect of age)
"""



# ╔═╡ 0dd0d060-81b4-4cc0-b306-dda7589adbd2
f11_Cohort =  @formula zScore ~ 1 + Test * a1 * Sex +  zerocorr(1 + Test | Cohort)

# ╔═╡ f68f6884-7c37-4868-b0b7-01ad450ee383
m11_Cohort = fit(MixedModel, f11_Cohort, dat)

# ╔═╡ ef1dfbf4-700c-4093-9824-01368c511531
VarCorr(m11_Cohort)

# ╔═╡ ace9a2f8-075a-49b5-94eb-12be4ffb6568
f11_School =  @formula zScore ~ 1 + Test * a1 * Sex +  (1 + Test + a1 | School)

# ╔═╡ 167fbd71-2a7b-4afa-8ffb-815dcda63129
m11_School = fit(MixedModel, f11_School, dat)

# ╔═╡ 68ee8abb-04e6-45ee-8f0a-033e7d2ef57b
VarCorr(m11_School)

# ╔═╡ dd8b5c05-c1c8-4778-b179-1ad39f50f8d3
f12 = @formula zScore ~ 1 + Test * a1 * Sex +  
                       (1 + Test + a1 | School) + zerocorr(1 + Test | Cohort)

# ╔═╡ 25376bb3-4673-4c89-8600-6e3da85971b4
m12 = fit(MixedModel, f12, dat)

# ╔═╡ e7cdd374-57c3-4d3c-9b1b-b6a211733b81
VarCorr(m12)

# ╔═╡ 5ffbc2ca-02df-4dd7-8e3b-f692d7d12adb


# ╔═╡ 1a2586a6-ff9d-4067-a5af-6c0ee6929e81
md"""
### 2.2 Conditional modes
"""

# ╔═╡ d2c0979e-ed52-4b56-b3ec-58db1609843e
cms = raneftables(m11_School)

# ╔═╡ 776cec2c-c566-4f40-81f5-12546c6a4da7
cms_Chrt = DataFrame(cms.School);

# ╔═╡ cf3f5312-6702-4069-95aa-862dc54d9a34
cvs = condVar(m11_School)

# ╔═╡ 10ef55d6-6048-4353-92e5-e4621bfb130e
md"""
### 3 MixedModelsMakie functions

Information on random effects conditional modes/means, variances, etc.
"""

# ╔═╡ cbe2197e-12c7-4d55-8817-d25d7b7cc5bd
struct RanefInfo{T<:AbstractFloat}
    cnames::Vector{String}
    levels::Vector
    ranef::Matrix{T}
    stddev::Matrix{T}
end

# ╔═╡ 32acfdf0-ecb6-4acb-904f-d7b963771d6d
md"""

ranefinfo(m::LinearMixedModel)

Return a `NamedTuple{fnames(m), NTuple(k, RanefInfo)}` from model `m`
"""

# ╔═╡ 90ce17cc-d497-4605-aa62-be69343e03eb
function ranefinfo(m::LinearMixedModel{T}) where {T}
    fn = fnames(m)
    val = sizehint!(RanefInfo[], length(fn))
    for (re, eff, cv) in zip(m.reterms, ranef(m), condVar(m))
        push!(
            val, 
            RanefInfo(
                re.cnames,
                re.levels,
                Matrix(adjoint(eff)),
                Matrix(adjoint(dropdims(sqrt.(mapslices(diag, cv, dims=1:2)), dims=2))),
            )
        )
    end
    NamedTuple{fn}((val...,))
end

# ╔═╡ cd4d8409-cb5e-4f38-bd63-c54b8f09e5ff
function caterpillar!(f::Figure, r::RanefInfo; orderby=1)
    rr = r.ranef
    vv = view(rr, :, orderby)
    ord = sortperm(vv)
    y = axes(rr, 1)
    cn = r.cnames
    axs = [Axis(f[1, j]) for j in axes(rr, 2)]
    linkyaxes!(axs...)
    for (j, ax) in enumerate(axs)
        xvals = view(rr, ord, j)
        scatter!(ax, xvals, y, color=(:red, 0.2))
        errorbars!(ax, xvals, y, view(r.stddev, ord, j), direction=:x)
        ax.xlabel = cn[j]
        ax.yticks = y
        j > 1 && hideydecorations!(ax, grid=false)
    end
    axs[1].yticks = (y, r.levels[ord])
    f
end

# ╔═╡ bd10bfa8-9e22-4927-b83e-eb80f328c6a3
md"""
    caterpillar(m::LinearMixedModel, gf::Symbol)

Returns a `Figure` of a "caterpillar plot" of the random-effects means and prediction intervals

A "caterpillar plot" is a horizontal error-bar plot of conditional means and standard deviations
of the random effects.
"""


# ╔═╡ 727059df-d7a5-4714-8dcb-6568d9119510
function caterpillar(m::LinearMixedModel, gf::Symbol=first(fnames(m)))
    caterpillar!(Figure(resolution=(1000,800)), ranefinfo(m)[gf])
end

# ╔═╡ 5a62529e-cc10-4e2c-8d75-125951f2dc34
md"""
  caterpillar!(f::Figure, r::RanefInfo; orderby=1)

Add Axes of a caterpillar plot from `r` to `f`.

The order of the levels on the vertical axes is increasing `orderby` column
of `r.ranef`, usually the `(Intercept)` random effects.
"""

# ╔═╡ a9043efa-17e5-498f-a744-7c0690446263
md"""
## 4. Caterpillar plots
### 4.1 LMM `m11_Cohort`
"""
  

# ╔═╡ 07a00ce2-4658-49ff-a384-4b5230691a7b
caterpillar(m11_Cohort)

# ╔═╡ 53ed436c-82b9-4331-bef4-0e034c233e7f
md"""
### 4.2 LMM `m11_School`
"""

# ╔═╡ 44b78423-e28c-4787-921e-e4ba4173a31c
caterpillar(m11_School)

# ╔═╡ 8e15a0b8-b445-4bec-8016-7b6eacf3c889
md"""
### 4.3 LMM `m12`
"""

# ╔═╡ 90dc96b9-3cc6-47d3-84d0-6c3eb8a05a5d
last(fnames(m12))

# ╔═╡ 12518738-2269-4019-9572-bfe6c9dc224f
caterpillar(m12)

# ╔═╡ Cell order:
# ╠═9396fcac-b0b6-11eb-3a60-9f2ce25df953
# ╟─eebef932-1d63-41e5-998f-9748379c43af
# ╠═4ecf1267-aac9-4e8f-a522-5595b1205f43
# ╠═faafd359-fb38-4f49-9dfd-19cb4f0c5a54
# ╟─ee85abd9-0172-44d3-b03a-c6a780d33c72
# ╠═39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
# ╟─0051ec83-7c30-4d28-9dfa-4c6f5d0259fa
# ╠═7440c439-9d4c-494f-9ac6-18c9fb2fe144
# ╠═16c3cdaa-49fa-46d5-a844-03094329fe4c
# ╠═64cc1f8e-f831-4a53-976f-dc7600b5634d
# ╟─c6c6f056-b8b9-4190-ac14-b900bafa04df
# ╠═c5326753-a03b-4739-a82e-90ffa7c1ebdb
# ╟─e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
# ╠═0dd0d060-81b4-4cc0-b306-dda7589adbd2
# ╠═f68f6884-7c37-4868-b0b7-01ad450ee383
# ╠═ef1dfbf4-700c-4093-9824-01368c511531
# ╠═ace9a2f8-075a-49b5-94eb-12be4ffb6568
# ╠═167fbd71-2a7b-4afa-8ffb-815dcda63129
# ╠═68ee8abb-04e6-45ee-8f0a-033e7d2ef57b
# ╠═dd8b5c05-c1c8-4778-b179-1ad39f50f8d3
# ╠═25376bb3-4673-4c89-8600-6e3da85971b4
# ╠═e7cdd374-57c3-4d3c-9b1b-b6a211733b81
# ╠═5ffbc2ca-02df-4dd7-8e3b-f692d7d12adb
# ╠═1a2586a6-ff9d-4067-a5af-6c0ee6929e81
# ╠═d2c0979e-ed52-4b56-b3ec-58db1609843e
# ╠═776cec2c-c566-4f40-81f5-12546c6a4da7
# ╠═cf3f5312-6702-4069-95aa-862dc54d9a34
# ╟─10ef55d6-6048-4353-92e5-e4621bfb130e
# ╠═cbe2197e-12c7-4d55-8817-d25d7b7cc5bd
# ╟─32acfdf0-ecb6-4acb-904f-d7b963771d6d
# ╠═90ce17cc-d497-4605-aa62-be69343e03eb
# ╠═cd4d8409-cb5e-4f38-bd63-c54b8f09e5ff
# ╟─bd10bfa8-9e22-4927-b83e-eb80f328c6a3
# ╠═727059df-d7a5-4714-8dcb-6568d9119510
# ╟─5a62529e-cc10-4e2c-8d75-125951f2dc34
# ╟─a9043efa-17e5-498f-a744-7c0690446263
# ╠═07a00ce2-4658-49ff-a384-4b5230691a7b
# ╟─53ed436c-82b9-4331-bef4-0e034c233e7f
# ╠═44b78423-e28c-4787-921e-e4ba4173a31c
# ╟─8e15a0b8-b445-4bec-8016-7b6eacf3c889
# ╠═90dc96b9-3cc6-47d3-84d0-6c3eb8a05a5d
# ╠═12518738-2269-4019-9572-bfe6c9dc224f
