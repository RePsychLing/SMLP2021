### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ d17d5cf8-988a-11eb-03dc-23f1449f5563
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ 6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
begin
	using AlgebraOfGraphics
	using AlgebraOfGraphics: linear
	using CairoMakie  # for Scatter
end


# ╔═╡ 40300e47-3092-43fd-9fb3-a28fa1df215a
begin
	using CSV, RCall, DataFrames, MixedModels
	using Statistics, StatsBase
	using CategoricalArrays, Arrow
end

# ╔═╡ 4cd88e2b-c6f0-49cb-846a-eaae9f892e02
md"""
# Mixed Models Tutorial: Contrast Coding

This script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

All children were between 6.0 and 6.99 years at legal keydate (30 September) of school enrollement, that is in their ninth year of life in the third grade. To avoid delays associated with model fitting we work with a reduced data set and less complex models than those in the reference publication. The script requires only a few changes to specify the more complex models in the paper. 

The script is structured in three main sections: 

1. **Setup** with reading and examing the data

2. **Contrasts coding**
+ Effect and seqential difference contrasts
+ Helmert contrast
+ Hypothesis contrast
+ PCA-based contrast

3. **Other topics**
+ LMM goodness of fit does not depend on contrast (i.e., reparameterization)
+ VCs and CPs depend on contrast
+ VCs and CPs depend on random factor
"""

# ╔═╡ be9aca0c-3e3d-4470-a399-3438cb020915
md"""
## 1. Setup
### 1.0 Packages and functions
"""

# ╔═╡ d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
begin
	function viewdf(x)
		DataFrame(Arrow.Table(take!(Arrow.write(IOBuffer(), x))))
	end
end


# ╔═╡ f710eadc-15e2-4911-b6b2-2b97d9ce7e3e
md"""
### 1.1 Readme for 'EmotikonSubset.rds'

1. Cohort: 9 levels; 2011-2019
2. School: 46 levels 
3. Child: 11,566 levels
4. Sex: 5,893 girls, 5,673 boys
5. age: test date - middle of month of birthdate 
6. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 7. score - see units

 ### 1.2 Preprocessing

 #### 1.2.1 Read data
"""

# ╔═╡ bea880a9-0513-4b70-8a72-af73fb83bd19
dat = rcopy(R"readRDS('./data/EmotikonSubset.rds')");

# ╔═╡ f51a7baf-a68e-4fae-b62d-9adfbf2b3412
md"""
#### 1.2.2 Transformations
"""

# ╔═╡ 924e86cc-d799-4ed9-ac89-34be0b4872ed
begin
	transform!(dat, :age, :age => (x -> x .- 8.5) => :a1); # centered age (linear)
	transform!(dat,  :a1, :a1  => (x -> x.^2) => :a2);     # centered age (quadr.)
	select!(groupby(dat,  :Test), :, :score => zscore => :zScore); # z-score
	viewdf(dat)
end

# ╔═╡ 8318ac96-92a7-414d-8611-f1830642539a
begin
	dat2 = combine(groupby(dat, [:Test, :Sex]), 
                             :score => mean, :score  => std, 
                             :zScore => mean, :zScore => std)
	viewdf(dat2)
end

# ╔═╡ fef952af-c3c0-4354-85a3-c5d5076d705c
md"""
#### 1.2.3 Figure of age x Sex x Test interactions
**<Still to be included>**
"""

# ╔═╡ 59afbe0c-0c70-4fc5-97f1-5020ef33b5cc
md"""
## 2. Contrast coding

Contrast coding is part of `StatsModels.jl`. Here is the primary author's (i.e., Dave Kleinschmidt's documentation of  [Modeling Categorical Data](https://juliastats.org/StatsModels.jl/stable/contrasts/#Modeling-categorical-data).

The random factors `Child`, `School`, and `Cohort` are assigned a _Grouping_ contrast. This contrast is needed when the number of groups (i.e., units, levels) is very large. This is the case for `Child` (i.e., the 108,925 children in the full and probably also the 11,566 children in the reduced data set). The assignment is not necessary for the typical sample size of experiments. However, we use this coding of random factors irrespective of the number of units associated with them to be transparent about the distinction between random and fixed factors.

A couple of general remarks about the following examples. First, all contrasts defined in this tutorial return an estimate of the _Grand Mean_ (GM) in the intercept, that is they are so-called sum-to-zero contrasts. In both `Julia` and `R` the default contrast is _Dummy_coding is not a sum-to-zero contrast, but returns the mean of the reference (control) group - unfortunately for (quasi-)experimentally minded scientists.

Second, The factor `Sex` has only two levels. We use _EffectCoding_ (also known as _Sum_ coding in `R`) to estimate the difference of the levels from the Grand Mean. Unlike in `R`, the default sign of the effect is for the second level (base is the first,  not the last level), but this can be changed with the `base` kwarg in the command. _Effect coding is a sum-to-zero contrast, but when applied to factors with more than two levels does not yield orthogonal contrasts. 

Finally, contrasts for the five levels of the fixed factor `Test` represent the hypotheses about differences between them. In this tutorial, we use this factor to illustrate various options. 

**We (initially) include only `Test` as fixed factor and `Child` as random factor. More complex LMMs can be specified by simply adding other fixed or random factors to the formula.**

### 2.1 _SeqDiffCoding_: `contr1`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise differences between the five neighboring levels of `Test`, that is: 

+ SDH1: 2-1
+ SDH2: 3-2
+ SDH3: 4-3
+ SDH4: 5-4

The levels were sorted such that these contrasts map onto four  _a priori_ hypotheses; in other words, they are _theoretically_ motivated pairwise comparisons. The motivation also encompasses theoretically motivated interactions with `Sex`. The order of levels can also be explicitly specified during contrast construction. This is very useful if levels are in a different order in the dataframe. We recommend it in general to increase transparency of the code. 

The statistical disadvantage of _SeqDiffCoding_ is that the contrasts are not orthogonal, that is the contrasts are correlated. This is obvious from the fact that levels 2, 3, and 4 are all used in two contrasts. One consequence of this is that correlation parameters estimated between neighboring contrasts (e.g., 2-1 and 3-2) are difficult to interpret. Usually, they will be negative because assuming some practical limitations on the overall range (e.g., between levels 1 and 3), a small "2-1" effect "correlates" negatively with a larger "3-2" effect for mathematical reasons. 

Obviously, the tradeoff between theoretical motivation and statistical purity is something that must be considered carefully when planning the analysis. 
"""


# ╔═╡ a7d67c88-a2b5-4326-af67-ae038fbaeb19
contr1 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	       Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"]))
	
	   )

# ╔═╡ e41bb345-bd0a-457d-8d57-1e27dde43a63
f_ovi1 = @formula zScore ~ 1 + Test + (1 | Child);

# ╔═╡ baf13d52-1569-4077-af4e-40aa53d38cf5
m_ovi1_SeqDiff = fit(MixedModel, f_ovi1, dat, contrasts=contr1)

# ╔═╡ d0b2bb2c-8e01-4c20-b902-e2ba5abae7cc
md"""
In this case, any differences between tests identified by the contrasts would be  spurious because each test was standardized (i.e., _M_=0, $SD$=1). The differences could also be due to an imbalance in the number of boys and girls or in the number of missing observations for each test. 

The primary interest in this study related to interactions of the test contrasts with and `age` and `Sex`. We start with age (linear) and its interaction with the four test contrasts.
"""

# ╔═╡ 8af679d3-e3f7-484c-b144-031354232388
begin
	f_ovi_2 = @formula zScore ~ 1 + Test*a1 + (1 | Child);
	m_ovi_SeqDiff_2 = fit(MixedModel, f_ovi_2, dat, contrasts=contr1)
end

# ╔═╡ bce94b8b-f633-42d1-972a-1f3f007dd818
md"""
The difference between older and younger childrend is larger for `Star_r` than for `Run` (0.2473).  `S20_r` did not differ significantly from `Star_r` (-0.0377) and `SLJ` (-0.0113) The largest difference in developmental gain was between `BPT` and `SLJ` (0.3355). 

P**lease note that standard errors of this LMM are anti-conservative because the LMM is missing a lot of information in the RES (e..g., contrast-related VCs snd CPs for `Child`, `School`, and `Cohort`.**

Next we add the main effect of `Sex` and its interaction with the four test contrasts. 
"""						

# ╔═╡ bb8a8f47-4f5b-4bc2-9b75-1c75927e3b92
begin
	f_ovi_3 = @formula zScore ~ 1 + Test*(a1+Sex) + (1 | Child);
	m_ovi_SeqDiff_3 = fit(MixedModel, f_ovi_3, dat, contrasts=contr1)
end

# ╔═╡ 424c7ed0-7f5b-4579-980f-989f7d43bc97
md"""
The significant interactions with `Sex` reflect mostly differences related to muscle power, where the physiological constitutions gives boys an advantage. The sex difference is smaller when coordination and cognition play a role -- as in the `Star_r` test. (Caveat: SEs are estimated with an underspecified RES.)

The final step in this first series is to add the interactions between the three covariates. A significant interaction between any of the four `Test` contrasts and age (linear) x `Sex` was hypothesized to reflect a prepubertal signal (i.e., hormones start to rise in girls' ninth year of life). However, this hypothesis is linked to a specific shape of the interaction: Girls would need to gain more than boys in tests of muscular power. 
"""

# ╔═╡ 9cb34e96-bf04-4d1a-962f-d98bb8429592
begin
	f_ovi = @formula zScore ~ 1 + Test*a1*Sex + (1 | Child);
	m_ovi_SeqDiff = fit(MixedModel, f_ovi, dat, contrasts=contr1)
end

# ╔═╡ 7de8eb1d-ce24-4e9f-a748-d0a266ea8428
md"""
The results are very clear: Despite an abundance of statistical power there is no evidence for the differences between boys and girls in how much the gain within the ninth year of life in these five physical fitness tests. The authors argue that in this case absence of evidence looks very much like evidence of absence of a hypothesized interaction. 

In the next two sections we use different contrasts. Does this have a bearing on this result?  We still ignore for now that we are looking at anti-conservative test statistics.
"""

# ╔═╡ e84051c6-61af-4c3a-a0c4-c1649c8fd2bf
md"""
### 2.2 _HelmertCoding_: `contr2`

The second set of contrasts uses _HelmertCoding_. Helmert coding codes each level as the difference from the average of the lower levels. With the default order of `Test` levels this yields the following test statistics which we describe in reverse order of appearance in model output

+ HeCH4: 5 - mean(1,2,3,4)
+ HeCH3: 4 - mean(1,2,3)
+ HeCH2: 3 - mean(1,2)
+ HeCH1: 2 - 1

In the model output, HeCH1 will be reported first and HeCH4 last.

There is some justification for the HeCH4 specification in a post-hoc manner because the fifth test (`BPT`) turned out to be different from the other four tests in that high performance is most likely also related to obesity, that is it might reflect physical unfitness. _A priori_ the SDH4 contrast 5-4 between `BPT` (5) and `SLJ` (4) was motivated because conceptually both are tests of the physical fitness component _Muscular Power_, `BPT` for upper limbs and `SLJ` for lower limbs, respectively.

One could argue that there is justification for HeCH3 because `Run` (1), `Star_r` (2), and `S20` (3) involve running but `SLJ` (4) does not. Sports scientists, however, recoil. For them it does not make much sense to average the different running tests, because they draw on completely different physiological resources; it is the old apples-and-oranges problem.

The justification for HeCH3 is that`Run` (1) and `Star_r` (2) draw more strongly on cardiosrespiratory _Endurance_ than `S20` (3) due to the longer duration of the runs compared to sprinting for 20 m which is a pure measure of the physical-fitness component _Speed_. Again, sports scientists are not very happy with this proposal.

Finally, HeCH1 contrasts the fitness components Endurance, indicated best by Run (1), and Coordination, indicated by `Star_r` (2). Endurance (i.e., running for 6 minutes) is considered to be the best indicator of health-related status among the five tests because it is a rather pure measure of cardiorespiratory fitness. The `Star_r` test requires execution of a pre-instructed sequence of forward, sideways, and backward runs. This coordination of body movements implies a demand on working memory (i.e., remembering the order of these subruns) and executive control processes, but it also requires endurance. HeCH1 yields a measure of Coordination "corrected" for the contribution of Endurance. 

The statistical advantage of _HelmertCoding_ is that the resulting contrasts are orthogonal (uncorrelated). This allows for optimal partitioning of variance and statistical power. It is also more efficient to estimate "orthogonal" than "non-orthogonal" random-effect structures.

"""

# ╔═╡ 7dca92f9-dcc2-462c-b501-9ecabce74005
contr2 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	       Dict(:Test => HelmertCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"]))
	   );

# ╔═╡ c8a80ac2-af56-4503-b05d-d727d7bcac12
m_ovi_Helmert = fit(MixedModel, f_ovi, dat, contrasts=contr2)

# ╔═╡ 9cfbcdf1-e78f-4e68-92b7-7761331d3972
md"""
We forego a detailed discussion of the effects, but note that again none of the interactions between `age x Sex` with the four test contrasts was significant.
"""

# ╔═╡ a272171a-0cda-477b-9fb1-32b249e1a2c8
md"""
### 2.3 _HypothesisCoding_: `contr3`

The third set of contrasts uses _HypothesisCoding_. _Hypothesis coding_ allows the user to specify their own _a priori_ contrast matrix, subject to the mathematical constraint that the matrix has full rank. For example, sport scientists agree that the first four tests can be contrasts with `BPT`,  because the difference is akin to a correction of overall physical fitness. However, they want to keep the pairwise comparisons for the first four tests. 

+ HyC1: `BPT` - mean(1,2,3,4)
+ HyC2: `Star_r` - `Run_r`
+ HyC3: `Run_r` - `S20_r`
+ HyC4: `S20_r` - `SLJ`
"""

# ╔═╡ 9f8a0809-0189-480b-957a-3d315763f8a4
contr3 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	       Dict(:Test => HypothesisCoding( [-1 -1 -1 -1 +4
	         							    -1 +1  0  0  0
	           								 0 -1 +1  0  0
	           								 0  0 -1 +1  0],
		                  levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"],
			      labels=["BPT-other", "Star-End", "S20-Star", "SLJ-S20"])));

# ╔═╡ 61f7ddca-27d0-4f4f-bac9-d757d19faa39
m_ovi_Hypo = fit(MixedModel, f_ovi, dat, contrasts=contr3)

# ╔═╡ f36bfe9b-4c7d-4e43-ae0f-56a4a3865c9c
md"""
The labeling of the contrasts is suboptimal. There is an option to generate one's own labels by extracting the indicator variables from the model matrix and giving them meaningful names. 

Anyway, none of the interactions between `age` x `Sex` with the four `Test` contrasts was significant for these contrasts.
"""

# ╔═╡ 7e34a00b-9504-4c13-a38e-cc9114930ca2
md"""
### 2.4 _PCA-based HypothesisCoding_: `contr4`

The fourth set of contrasts uses _HypothesisCoding_ to specify the set of contrasts implementing the loadings of the four principle components of the published LMM base on the test scores, not the test effects - coarse-grained, that is roughly according to their signs. This is actually a very interesting and plausible solution nobody proposed _a priori_. 

+ PCA1: BPT - `Run_r` 
+ PCA2: (`Star_r` + `S20_r` + `SLJ`) - (`BPT` + `Run_r`)
+ PCA3:  `Star_r` - (`S20_r` + `SLJ`)
+ PCA4:  `S20_r` - `SLJ`

PCA1 contrasts the worst and the best indicator of physical **health**; PCA2 contrasts these two against the core indicators of **physical fitness**; PCA3 tests within the core set the cognitive and the physical tests; and PCA4, finally, contrasts two types of lower muscular fitness, that is speed and power.
"""

# ╔═╡ 42984738-dbb3-47a8-bb6e-3a721edef16d
contr4 = merge(
		   Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		   Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	       Dict(:Test => HypothesisCoding( [-1  0  0  0 +1
	         							    -3 +2 +2 +2 -3
	           								 0 +2 -1 -1  0
	           								 0  0 +1 -1  0],
		                  levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"],
			              labels=["c5.1", "c234.15", "c2.34", "c3.4"])));

# ╔═╡ 6c51fa65-ea7d-46b5-b25b-8af43da4c1e8
m_ovi_PCA = fit(MixedModel, f_ovi, dat, contrasts=contr4)

# ╔═╡ 7eb0443f-eb43-4664-aeda-1bb254fdf241
md"""
Aside from the unfortunate labeling, there is actually an interaction with a z-value > 2.0 for the first PCA (i.e., `BPT` - `Run_r`).  This interaction would really need to be replicated to be taken seriously. It is probably be due to larger "unfitness" gains in boys that girls (i.e., in `BPT`)  relative to the slightly larger health-related "fitness" gains of girls than boys (i.e., in `Run_r`). 

**We will look at the PCA solutions in some detail in a separate tutorial.**
"""

# ╔═╡ 830b5655-50c0-425c-a8de-02743867b9c9
md"""
## 3. Other topics

### 3.1 Contrasts are re-parameterizations of the same model

The choice of contrast does not affect the model objective, in other words, they  all yield the same goodness of fit. It does not matter whether a contrast is orthogonal or not. 
"""

# ╔═╡ 6492eb08-239f-42cd-899e-9e227999d9e3
[objective(m_ovi_SeqDiff), objective(m_ovi_Helmert), objective(m_ovi_Hypo)]


# ╔═╡ 74d7a701-91cf-4dd1-9991-f94e156c3291
md"""
### 3.2 VCs and CPs depend on contrast coding

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
### 3.3 VCs and CPs depend on random factor
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

# ╔═╡ e84d2f33-ab76-4cd9-989c-0d27a22907ce
md"""
## 5. That's it
That's it for this tutorial. It is time to try you own contrast coding. You can use these data; there are many alternatives to set up hypotheses for the five tests. Of course and even better, code up some contrasts for data of your own.

Have fun!
"""

# ╔═╡ Cell order:
# ╠═d17d5cf8-988a-11eb-03dc-23f1449f5563
# ╟─6ef1b0be-3f8e-4fbf-b313-004c3ba001bd
# ╟─4cd88e2b-c6f0-49cb-846a-eaae9f892e02
# ╟─be9aca0c-3e3d-4470-a399-3438cb020915
# ╠═40300e47-3092-43fd-9fb3-a28fa1df215a
# ╠═d7ea6e71-b609-4a3c-8de6-b05e6cb0aceb
# ╟─f710eadc-15e2-4911-b6b2-2b97d9ce7e3e
# ╠═bea880a9-0513-4b70-8a72-af73fb83bd19
# ╟─f51a7baf-a68e-4fae-b62d-9adfbf2b3412
# ╠═924e86cc-d799-4ed9-ac89-34be0b4872ed
# ╠═8318ac96-92a7-414d-8611-f1830642539a
# ╟─fef952af-c3c0-4354-85a3-c5d5076d705c
# ╟─59afbe0c-0c70-4fc5-97f1-5020ef33b5cc
# ╠═a7d67c88-a2b5-4326-af67-ae038fbaeb19
# ╠═e41bb345-bd0a-457d-8d57-1e27dde43a63
# ╠═baf13d52-1569-4077-af4e-40aa53d38cf5
# ╟─d0b2bb2c-8e01-4c20-b902-e2ba5abae7cc
# ╠═8af679d3-e3f7-484c-b144-031354232388
# ╟─bce94b8b-f633-42d1-972a-1f3f007dd818
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
# ╟─7e34a00b-9504-4c13-a38e-cc9114930ca2
# ╠═42984738-dbb3-47a8-bb6e-3a721edef16d
# ╠═6c51fa65-ea7d-46b5-b25b-8af43da4c1e8
# ╟─7eb0443f-eb43-4664-aeda-1bb254fdf241
# ╟─830b5655-50c0-425c-a8de-02743867b9c9
# ╟─6492eb08-239f-42cd-899e-9e227999d9e3
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
# ╟─e84d2f33-ab76-4cd9-989c-0d27a22907ce
