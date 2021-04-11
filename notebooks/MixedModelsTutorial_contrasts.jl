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
	using Arrow, DataAPI, DataFrames, MixedModels
	using DataAPI: levels
end

# ╔═╡ 40300e47-3092-43fd-9fb3-a28fa1df215a
using ImageIO, Images, ImageMagick

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

# ╔═╡ 924e86cc-d799-4ed9-ac89-34be0b4872ed
md"""
## Plot the main results

We need to set up Pluto to display imgages.
"""

# ╔═╡ c3adec52-87b3-4655-9b5e-1936fcfb2877
md"""
The main results of relevance for this tutorial are shown in this figure from Fühner et al. (2021, Figure 2). There are developmental gains within the ninth year of life for each of the five tests and the tests differ in the magnitude of these gains. Also boys outperform girls on each test and, again, the sex difference varies across test.
"""

# ╔═╡ 8318ac96-92a7-414d-8611-f1830642539a
begin
	fig1 = load("./figures/Figure_2.png");
	fig1
end

# ╔═╡ 5096682d-ac64-42ca-aab0-6c24d8cde711
md"""
_Figure 1._ Performance differences between 8.0 und 9.0 years by sex in the five physical fitness tests presented as z-transformed data computed separately for each test. Endurance = cardiorespiratory endurance (i.e., 6 min run test), Coordination = star run test, Speed = 20-m linear sprint test, PowerLOW = power of lower limbs (i.e., standing long jump test), PowerUP = power of upper limbs (i.e., ball push test), SD = standard deviation. Points are binned observed child means; lines are simple regression fits to the observations; 95% confidence intervals for means ≈ 0.05 are not visible.
"""

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
f_ovi1 = @formula zScore ~ 1 + Test + (1 | Child);

# ╔═╡ baf13d52-1569-4077-af4e-40aa53d38cf5
m_ovi1_SeqDiff = fit(MixedModel, f_ovi1, dat, contrasts=contr1)

# ╔═╡ d0b2bb2c-8e01-4c20-b902-e2ba5abae7cc
md"""
The differences between tests identified by the contrasts are actually spurious because all tests were standardized to a mean of zero. The differences are due to an imbalance with respect to the number of boys and girls and the number of observations for each test. Moreover, given the large sample of 108,925 children, the authors had set the level of significance to |z| = 3.0.

The primary interest in this study relates to interactions of these test contrasts with `Sex` and `age`. In other words, are the differences between tests different for boys and girls and are they different for the young (< 8.5 years) and older (>= 8.5) children who are all in their ninth year of life. The age effect is estimated as the gain between 8.0 and 9.0 years of age.  

First, we add Sex and its interaction with the four test contrasts: 
"""

# ╔═╡ 8af679d3-e3f7-484c-b144-031354232388
begin
	f_ovi_2 = @formula zScore ~ 1 + Test*Sex + (1 | Child);
	m_ovi_SeqDiff_2 = fit(MixedModel, f_ovi_2, dat, contrasts=contr1)
end

# ╔═╡ 6b2bf9ad-1037-4261-876d-5f08b392a88c
md"""
The difference between boys and girls is larger for `Run` and `S20_r` than for  `Star_r` (i.e., Est. = -.1199 and .0388), respectively. Boys outperform girls more on `SLJ` than on `S20_r` (0.0348) as well as on `BPT` compared to `SLJ` (0.1444). 

We note that the standard errors are anti-conservative because the LMM is missing a lot of information in the RES (e..g., contrast-related VCs snd CPs for `Child`, `School`, and `Cohort`.

Next we add the linear cross-sectional age-effect and its interactions with the four test contrasts.
"""

# ╔═╡ bb8a8f47-4f5b-4bc2-9b75-1c75927e3b92
begin
	f_ovi_3 = @formula zScore ~ 1 + Test*(Sex+(age - 8.5)) + (1 | Child);
	m_ovi_SeqDiff_3 = fit(MixedModel, f_ovi_3, dat, contrasts=contr1)
end

# ╔═╡ 424c7ed0-7f5b-4579-980f-989f7d43bc97
md"""
The difference between older and younger childrend is larger for `Star_r` than for `Run` (0.2405) and  `Star_r` and  `S20_r` did not differ significantly (-0.0242).  . The gain was larger for `S20_r` than for `SLJ` (-0.0640) The largest difference in developmental gain was between `BPT` and `SLJ` (0.3379). 

Again, please note that the standard errors are anti-conservative because the LMM is missing a lot of information in the RES (e..g., contrast-related VCs snd CPs for `Child`, `School`, and `Cohort`.

In a final step, we add the interaction of the four test contrasts with the interaction of `Sex` and the linear effect of `age`. Such interaction would indicate that boys and girls develop differently for different tests. The authors hypothesized that physical fitness might pick up a prepubertal signal (i.e., sexual hormones start to rise in the ninth year of girls' lives,  but not yet boys') and that this would lead to larger developmental gain for girls than boys.
"""

# ╔═╡ 9cb34e96-bf04-4d1a-962f-d98bb8429592
begin
	f_ovi = @formula zScore ~ 1 + Test*Sex*(age - 8.5) + (1 | Child);
	m_ovi_SeqDiff = fit(MixedModel, f_ovi, dat, contrasts=contr1)
end

# ╔═╡ 7de8eb1d-ce24-4e9f-a748-d0a266ea8428
md"""
The results are very clear: Despite an abundance of statistical power there is no evidence for the differences between boys and girls in how much the gain within the ninth year of life in these five physical fitness tests. The authors argue that in this case absence of evidence looks very much like evidence of absence of a hypothesized interaction. 

In the next two sections we use different contrasts. Does this have a bearing on this result?  We still ignore for now that we are looking at anti-conservative test statistics.
"""

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

# ╔═╡ 9cfbcdf1-e78f-4e68-92b7-7761331d3972
md"""
We forgo a detail re-statement of the effects, but note that again none of the interactions between `age x Sex` with the four test contrasts was significant.
"""

# ╔═╡ a272171a-0cda-477b-9fb1-32b249e1a2c8
md"""
### _HypothesisCoding_: `contr3`

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

# ╔═╡ f36bfe9b-4c7d-4e43-ae0f-56a4a3865c9c
md""""
None of the interactions between age x Sex with the four test contrasts was significant for these contrasts.
"""

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
	f_Child = @formula zScore ~ 1 + Test*Sex*(age - 8.5) + (1 + Test | Child);
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

`Sex` and `age` vary within `School`. Therefore, we also include their VCs and CPs in this model.
"""

# ╔═╡ 9f1d9769-aab3-4e7f-9dc0-6032361d279f
begin
	f_School = @formula zScore ~  1 + Test*Sex*(age - 8.5) + (1 + Test + Sex + (age -8.5) | School);
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

# ╔═╡ 4cdaecad-e0b4-400f-ae11-b5cbba101166
md"""
## Fine tuning the RES for `School`
The VCs and CPs for `Sex` are comparatively small. Do we need them? We check.
"""

# ╔═╡ 4ac9dd5c-d601-4e00-8ad1-cc9ce71ce6cf
begin
	f_School_nosex = @formula zScore ~  1 + Test*Sex*(age - 8.5) + (1 + Test + (age -8.5) | School);
	m_School_SeqDiff_nosex = fit(MixedModel, f_School_nosex, dat, contrasts=contr1);
	VarCorr(m_School_SeqDiff)
	MixedModels.likelihoodratiotest(m_School_SeqDiff, m_School_SeqDiff_nosex)
end

# ╔═╡ 11684c80-f816-4d02-a213-0710934334ae
md""" Ok, we do. How about just keeping the VC and dropping the CPs?"""

# ╔═╡ ef535d84-c720-46e1-b8cc-424c45506d9c
begin
	f_School_nosexCP = @formula zScore ~  1 + Test*Sex*(age - 8.5) + (1 + Test + (age -8.5) | School) + (0 + Sex | School);
	m_School_SeqDiff_nosexCP = fit(MixedModel, f_School_nosexCP, dat, contrasts=contr1);
	VarCorr(m_School_SeqDiff)
	MixedModels.likelihoodratiotest(m_School_SeqDiff, m_School_SeqDiff_nosexCP)
end

# ╔═╡ f50733eb-653f-4a93-a401-cb65ec0ccb41
md"""Clearly, with sex-relatd CPs for `School` we are only fitting noise.

This is what the parsimonious RES looks like.
"""

# ╔═╡ 27fd45a5-608d-41e3-bd30-fb974e1368cc
	VarCorr(m_School_SeqDiff_nosexCP)

# ╔═╡ e84d2f33-ab76-4cd9-989c-0d27a22907ce
md"""
That's it for this tutorial. It is time to try you own contrast coding. You can use these data; there are many alternatives to set up hypotheses for the five tests. Of course and even better, code up some contrasts for data of your own.

Have fun!
"""

# ╔═╡ Cell order:
# ╠═d17d5cf8-988a-11eb-03dc-23f1449f5563
# ╟─6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
# ╠═4cd88e2b-c6f0-49cb-846a-eaae9f892e02
# ╠═d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
# ╟─f710eadc-15e2-4911-b6b2-2b97d9ce7e3e
# ╠═bea880a9-0513-4b70-8a72-af73fb83bd19
# ╠═f51a7baf-a68e-4fae-b62d-9adfbf2b3412
# ╠═924e86cc-d799-4ed9-ac89-34be0b4872ed
# ╠═40300e47-3092-43fd-9fb3-a28fa1df215a
# ╟─c3adec52-87b3-4655-9b5e-1936fcfb2877
# ╠═8318ac96-92a7-414d-8611-f1830642539a
# ╟─5096682d-ac64-42ca-aab0-6c24d8cde711
# ╟─59afbe0c-0c70-4fc5-97f1-5020ef33b5cc
# ╠═a7d67c88-a2b5-4326-af67-ae038fbaeb19
# ╠═e41bb345-bd0a-457d-8d57-1e27dde43a63
# ╠═baf13d52-1569-4077-af4e-40aa53d38cf5
# ╟─d0b2bb2c-8e01-4c20-b902-e2ba5abae7cc
# ╠═8af679d3-e3f7-484c-b144-031354232388
# ╟─6b2bf9ad-1037-4261-876d-5f08b392a88c
# ╠═bb8a8f47-4f5b-4bc2-9b75-1c75927e3b92
# ╟─424c7ed0-7f5b-4579-980f-989f7d43bc97
# ╠═9cb34e96-bf04-4d1a-962f-d98bb8429592
# ╟─7de8eb1d-ce24-4e9f-a748-d0a266ea8428
# ╟─e84051c6-61af-4c3a-a0c4-c1649c8fd2bf
# ╠═7dca92f9-dcc2-462c-b501-9ecabce74005
# ╠═c8a80ac2-af56-4503-b05d-d727d7bcac12
# ╟─9cfbcdf1-e78f-4e68-92b7-7761331d3972
# ╟─a272171a-0cda-477b-9fb1-32b249e1a2c8
# ╠═9f8a0809-0189-480b-957a-3d315763f8a4
# ╠═61f7ddca-27d0-4f4f-bac9-d757d19faa39
# ╟─f36bfe9b-4c7d-4e43-ae0f-56a4a3865c9c
# ╟─830b5655-50c0-425c-a8de-02743867b9c9
# ╠═6492eb08-239f-42cd-899e-9e227999d9e3
# ╟─74d7a701-91cf-4dd1-9991-f94e156c3291
# ╠═61e9e6c3-31f5-4037-b0f0-40b82d1d8fdc
# ╠═bbeed889-dc08-429e-9211-92e7be35ec97
# ╠═f593f70b-ec70-4e96-8246-f1b8203199e0
# ╠═2ab82d40-e65a-4801-b118-6cfe340708c0
# ╟─65c10c72-cbcf-4923-bdc2-a23aa6ddd856
# ╠═9f1d9769-aab3-4e7f-9dc0-6032361d279f
# ╠═cbd98fc6-4d7e-42f0-84d0-525ab947187a
# ╠═d8bcb024-a971-47d6-a3f3-9eff6f3a9386
# ╠═c47058ac-31e6-4b65-8116-e313049f6323
# ╟─4cdaecad-e0b4-400f-ae11-b5cbba101166
# ╠═4ac9dd5c-d601-4e00-8ad1-cc9ce71ce6cf
# ╟─11684c80-f816-4d02-a213-0710934334ae
# ╠═ef535d84-c720-46e1-b8cc-424c45506d9c
# ╟─f50733eb-653f-4a93-a401-cb65ec0ccb41
# ╠═27fd45a5-608d-41e3-bd30-fb974e1368cc
# ╟─e84d2f33-ab76-4cd9-989c-0d27a22907ce
