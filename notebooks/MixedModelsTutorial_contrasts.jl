### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ d17d5cf8-988a-11eb-03dc-23f1449f5563
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 4cd88e2b-c6f0-49cb-846a-eaae9f892e02
begin
	using Arrow, DataAPI, DataFrames, MixedModels, StatsBase
	using DataAPI: levels
end

# ╔═╡ 6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
md"""
# Mixed Models Tutorial: Contrast Coding

Ths script uses data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools. 

## Readme for 'fggk21.rds'

1. Cohort: 9 levels; 2011-2019
2. District ("Landkreis"): 18 levels
3. BU: 2 levels; school is close ("B") or far ("U") from Berlin
4. School: 515 levels 
5. Child: 108295 levels; all children were between 8.0 and 8.99 years at legal keydate (30 September).
6. Sex:
7. age: test date - middle of month of birthdate 
8. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 9. score - see units
 10. zScore - this variable will be used as DV for comparison across tests

## Read and examine data
"""

# ╔═╡ d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
dat = Arrow.Table("./data/fggk21.arrow");


# ╔═╡ f710eadc-15e2-4911-b6b2-2b97d9ce7e3e
describe(DataFrame(dat))	

# ╔═╡ bea880a9-0513-4b70-8a72-af73fb83bd19
levels(dat.Test)

# ╔═╡ f51a7baf-a68e-4fae-b62d-9adfbf2b3412
levels(dat.Sex)

# ╔═╡ d5a79b7a-10d4-4fec-9952-42313c220363
md"""Test levels are not in order; Sex labels should be changed, too."""

# ╔═╡ 59afbe0c-0c70-4fc5-97f1-5020ef33b5cc
md"""
## Contrast coding

Contrast coding is part of StatsModels.jl. The website for the documentation is https://juliastats.org/StatsModels.jl/stable/contrasts/#Modeling-categorical-data

For random factors `Child`, `School`, and `Cohort` are assigned a _Grouping_ contrast. This contrast is needed when the number of groups (i.e., units, levels) is very large. This is the case for `Child` (108,925) and `School` (515).

The factor `Sex` has only two levels. We use _EffectCoding_ (also known as _Sum_ coding) to estimate the difference of the levels from the Grand Mean. Unlike in R, the default sign of the effect is for the second level (base is the first,  not the last level), but this can be changed with the `base` kwarg in the command. Note that _SumCoding_ applied to factors with more than two levels yields non-orthogonal contrasts. 

Finally, contrasts for the five levels of the fixed factor `Test` represent the hypotheses about differences between them. In this tutorial, we use this factor to illustrate various options. 

All contrasts defined in this tutorial return an estimate of the _Grand Mean_ (GM) in the intercept. 

**In the following examples we initially include only `Test` as fixed and `Child` as random factor`. More complex LMMs can be specified by simply adding other fixed or random factors to the formula.**`

### _SeqDiffCoding_: `contr1`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise differences between the five neighboring levels of `Test`, that is: 

+ SDH1: 2-1
+ SDH2: 3-2
+ SDH3: 4-3
+ SDH4: 5-4

The levels were sorted such that these contrasts map onto four  _a priori_ hypotheses; in other words, they are _theoretically_ motivated pairwise comparisons. The motivation also encompasses theoretically motivated interactions with `Sex`. The order of levels can also be explicitly specified during contrast construction. This is very useful if levels are in a different order in the dataframe.

The statistical disadvantage of _SeqDiffCoding_ is that the contrasts are not orthogonal, that is the contrasts are correlated. This is obvious from the fact that levels 2, 3, and 4 are all used in two contrasts. One consequence of this is that correlation parameters estimated between neighboring contrasts (e.g., 2-1 and 3-2) are difficult to interpret. Usually, they will be negative because assuming some practical limitations on the overall range (e.g., between levels 1 and 3), a small "2-1" effect "correlates" negatively with a larger "3-2" effect for mathematical reasons. 

Obviously, the tradeoff between theoretical motivation and statistical purity is something that must be considered carefully when planning the analysis. 
"""


# ╔═╡ a7d67c88-a2b5-4326-af67-ae038fbaeb19
contr1 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["female", "male"])),
	       Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"]))
	
	   )

# ╔═╡ e41bb345-bd0a-457d-8d57-1e27dde43a63
f_ovi = @formula zScore ~ 1 + Test + (1 | Child);

# ╔═╡ baf13d52-1569-4077-af4e-40aa53d38cf5
m_ovi_SeqDiff = fit(MixedModel, f_ovi, dat, contrasts=contr1)

# ╔═╡ e84051c6-61af-4c3a-a0c4-c1649c8fd2bf
md"""
### _HelmertCoding_: `contr2`

The second set of contrasts uses _HelmertCoding_. Helmert coding codes each level as the difference from the average of the lower levels. With the default order of `Test` levels this yields the following test statistics which we describe in reverse order of appearance in model output

+ HCH4: 5 - mean(1,2,3,4)
+ HCH3: 4 - mean(1,2,3)
+ HCH2: 3 - mean(1,2)
+ HCH1: 2 - 1

In the model output, HCH1 will be reported first and HCH4 last.

There is some justification for the HCH4 specification in a post-hoc manner because the fifth test (`BPT`) turned out to be different from the other four tests in that high performance is most likely also related to obesity, that is it might reflect physical unfitness. _A priori_ the SDH4 contrast 5-4 between `BPT` (5) and `SLJ` (4) was motivated because conceptually both are tests of the physical fitness component _Muscular Power_, `BPT` for upper limbs and `SLJ` for lower limbs, respectively.

One could argue that there is justification for HCH3 because `Run` (1), `Star_r` (2), and `S20` (3) involve running but `SLJ` (4) does not. Sports scientists, however, recoil. For them it does not make much sense to average the different running tests, because they draw on completely different physiological resources; it is the old apples-and-oranges problem.

The justification for HCH3 is that`Run` (1) and `Star_r` (2) draw more strongly on cardiosrespiratory _Endurance_ than `S20` (3) due to the longer duration of the runs compared to sprinting for 20 m which is a pure measure of the physical-fitness component _Speed_. Again, sports scientists are not very happy with this proposal.

Finally, HCH1 contrasts the fitness components Endurance, indicated best by Run (1), and Coordination, indicated by `Star_r` (2). Endurance (i.e., running for 6 minutes) is considered to be the best indicator of health-related status among the five tests because it is a rather pure measure of cardiorespiratory fitness. The `Star_r` test requires execution of a pre-instructed sequence of forward, sideways, and backward runs. This coordination of body movements implies a demand on working memory (i.e., remembering the order of these subruns) and executive control processes, but it also requires endurance. HCH1 yields a measure of Coordination "corrected" for the contribution of Endurance. 

The statistical advantage of _HelmertCoding_ is that the resulting contrasts are orthogonal (uncorrelated). This allows for optimal partitioning of variance and statistical power. It is also more efficient to estimate "orthogonal" than "non-orthogonal" random-effect structures.

"""

# ╔═╡ 7dca92f9-dcc2-462c-b501-9ecabce74005
contr2 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["female", "male"])),
	       Dict(:Test => HelmertCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"]))
	
	   );

# ╔═╡ c8a80ac2-af56-4503-b05d-d727d7bcac12
m_ovi_Helmert = fit(MixedModel, f_ovi, dat, contrasts=contr2)

# ╔═╡ a272171a-0cda-477b-9fb1-32b249e1a2c8
md"""
### Special _HypothesisCoding_: `contr3`

The third set of contrasts uses _HypothesisCoding_. _Hypothesis coding_ allows the user to specify their own _a priori_ contrast matrix, subject to the mathematical constraint that the matrix has full rank. For example, sport scientists agree that the first four tests can be contrasts with `BPT`,  because the difference is akin to a correction of overall physical fitness. However, they want to keep the pairwise comparisons for the first four tests. 

+ HC1: 5 - mean(1,2,3,4)
+ HC2: 2-1
+ HC3: 3-2
+ HC4: 4-3

"""

# ╔═╡ 9f8a0809-0189-480b-957a-3d315763f8a4
contr3 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["female", "male"])),
	       Dict(:Test => HypothesisCoding( [-1 -1 -1 -1 +4
	         							    -1 +1  0  0  0
	           								 0 -1 +1  0  0
	           								 0  0 -1 +1  0],
		                  levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"])));

# ╔═╡ 61f7ddca-27d0-4f4f-bac9-d757d19faa39
m_ovi_Hypo = fit(MixedModel, f_ovi, dat, contrasts=contr3)

# ╔═╡ 830b5655-50c0-425c-a8de-02743867b9c9
md"""
## Other topics

### Contrasts are re-parameterizations of the same model

The choice of contrast does not affect the model objective, in other words, they  all yield the same goodness of fit. It does not matter whether a contrast is orthogonal or not. 
"""

# ╔═╡ 6492eb08-239f-42cd-899e-9e227999d9e3
[objective(m_ovi_SeqDiff), objective(m_ovi_Helmert), objective(m_ovi_Hypo)]


# ╔═╡ 74d7a701-91cf-4dd1-9991-f94e156c3291
md"""
### VCs and CPs depend on contrast coding

Trivially, the meaning of a contrast depends on its definition. Consequently, the contrast specification has a big effect on the random-effect structure. As an illustration, we refit the three LMMs with variance components (VCs) and correlation parameters (CPs) for `Child`-related contrasts of `Test`. 
"""

# ╔═╡ 61e9e6c3-31f5-4037-b0f0-40b82d1d8fdc
begin
	f_Child = @formula zScore ~ 1 + Test + (1 + Test | Child);
	m_Child_SeqDiff = fit(MixedModel, f_Child, dat, contrasts=contr1);
	m_Child_Helmert = fit(MixedModel, f_Child, dat, contrasts=contr2);
	m_Child_Hypo = fit(MixedModel, f_Child, dat, contrasts=contr3);
end

# ╔═╡ bbeed889-dc08-429e-9211-92e7be35ec97
VarCorr(m_Child_SeqDiff)

# ╔═╡ f593f70b-ec70-4e96-8246-f1b8203199e0
VarCorr(m_Child_Helmert)

# ╔═╡ 2ab82d40-e65a-4801-b118-6cfe340708c0
VarCorr(m_Child_Hypo)

# ╔═╡ 65c10c72-cbcf-4923-bdc2-a23aa6ddd856
md"""
### VCs and CPs depend on random factor
VCs and CPs resulting from a set of test contrasts can also be estimated for the random factor `School`. Of course, these VCs and CPs may look different from the ones we just estimated for `Child`.
"""

# ╔═╡ 9f1d9769-aab3-4e7f-9dc0-6032361d279f
begin
	f_School = @formula zScore ~ 1 + Test + (1 + Test | School);
	m_School_SeqDiff = fit(MixedModel, f_School, dat, contrasts=contr1);
	m_School_Helmert = fit(MixedModel, f_School, dat, contrasts=contr2);
	m_School_Hypo = fit(MixedModel, f_School, dat, contrasts=contr3);
end

# ╔═╡ cbd98fc6-4d7e-42f0-84d0-525ab947187a
VarCorr(m_School_SeqDiff)

# ╔═╡ d8bcb024-a971-47d6-a3f3-9eff6f3a9386
VarCorr(m_School_Helmert)

# ╔═╡ c47058ac-31e6-4b65-8116-e313049f6323
VarCorr(m_School_Hypo)

# ╔═╡ Cell order:
# ╠═d17d5cf8-988a-11eb-03dc-23f1449f5563
# ╟─6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
# ╠═4cd88e2b-c6f0-49cb-846a-eaae9f892e02
# ╠═d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
# ╟─f710eadc-15e2-4911-b6b2-2b97d9ce7e3e
# ╠═bea880a9-0513-4b70-8a72-af73fb83bd19
# ╠═f51a7baf-a68e-4fae-b62d-9adfbf2b3412
# ╟─d5a79b7a-10d4-4fec-9952-42313c220363
# ╠═59afbe0c-0c70-4fc5-97f1-5020ef33b5cc
# ╠═a7d67c88-a2b5-4326-af67-ae038fbaeb19
# ╠═e41bb345-bd0a-457d-8d57-1e27dde43a63
# ╠═baf13d52-1569-4077-af4e-40aa53d38cf5
# ╟─e84051c6-61af-4c3a-a0c4-c1649c8fd2bf
# ╠═7dca92f9-dcc2-462c-b501-9ecabce74005
# ╠═c8a80ac2-af56-4503-b05d-d727d7bcac12
# ╟─a272171a-0cda-477b-9fb1-32b249e1a2c8
# ╠═9f8a0809-0189-480b-957a-3d315763f8a4
# ╠═61f7ddca-27d0-4f4f-bac9-d757d19faa39
# ╠═830b5655-50c0-425c-a8de-02743867b9c9
# ╠═6492eb08-239f-42cd-899e-9e227999d9e3
# ╠═74d7a701-91cf-4dd1-9991-f94e156c3291
# ╠═61e9e6c3-31f5-4037-b0f0-40b82d1d8fdc
# ╠═bbeed889-dc08-429e-9211-92e7be35ec97
# ╠═f593f70b-ec70-4e96-8246-f1b8203199e0
# ╠═2ab82d40-e65a-4801-b118-6cfe340708c0
# ╠═65c10c72-cbcf-4923-bdc2-a23aa6ddd856
# ╠═9f1d9769-aab3-4e7f-9dc0-6032361d279f
# ╠═cbd98fc6-4d7e-42f0-84d0-525ab947187a
# ╠═d8bcb024-a971-47d6-a3f3-9eff6f3a9386
# ╠═c47058ac-31e6-4b65-8116-e313049f6323
