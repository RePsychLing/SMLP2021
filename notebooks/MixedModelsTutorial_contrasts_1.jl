### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 40300e47-3092-43fd-9fb3-a28fa1df215a
begin
	using Arrow
	using CSV
	using DataFrames
	using MixedModels
	using RCall 
	using Statistics
	using StatsBase
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
function viewdf(x)
	DataFrame(Arrow.Table(take!(Arrow.write(IOBuffer(), x))))
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

+ SDC1: 2-1
+ SDC2: 3-2
+ SDC3: 4-3
+ SDC4: 5-4

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

+ HeC4: 5 - mean(1,2,3,4)
+ HeC3: 4 - mean(1,2,3)
+ HeC2: 3 - mean(1,2)
+ HeC1: 2 - 1

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
	 Dict(:Sex =>     EffectsCoding( levels=["Girls", "Boys"])),
	 Dict(:Test => HelmertCoding(
			       levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"])));

# ╔═╡ c8a80ac2-af56-4503-b05d-d727d7bcac12
m_ovi_Helmert = fit(MixedModel, f_ovi, dat, contrasts=contr2)

# ╔═╡ 9cfbcdf1-e78f-4e68-92b7-7761331d3972
md"""
We forego a detailed discussion of the effects, but note that again none of the interactions between `age x Sex` with the four test contrasts was significant. 

The default labeling of Helmert contrasts may lead to confusions with other contrasts. Therefore, we could provide our own labels:

`labels=["c2.1", "c3.12", "c4.123", "c5.1234]"`

Once the order of levels is memorized the solution offered here is quite transparent.  
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
With HyppthesisCoding we must generate our own labels for the contrasts. The default labeling of contrasts is usually not interpretable and we provide our own. 

Anyway, none of the interactions between `age` x `Sex` with the four `Test` contrasts was significant for these contrasts.
"""

# ╔═╡ 7e34a00b-9504-4c13-a38e-cc9114930ca2
md"""
### 2.4 _PCA-based HypothesisCoding_: `contr4`

The fourth set of contrasts uses _HypothesisCoding_ to specify the set of contrasts implementing the loadings of the four principle components of the published LMM base on the test scores, not the test effects - coarse-grained, that is roughly according to their signs. This is actually a very interesting and plausible solution nobody proposed _a priori_. 

+ PC1: `BPT` - `Run_r` 
+ PC2: (`Star_r` + `S20_r` + `SLJ`) - (`BPT` + `Run_r`)
+ PC3:  `Star_r` - (`S20_r` + `SLJ`)
+ PC4:  `S20_r` - `SLJ`

PC1 contrasts the worst and the best indicator of physical **health**; PC2 contrasts these two against the core indicators of **physical fitness**; PC3 tests within the core set the cognitive and the physical tests; and PC4, finally, contrasts two types of lower muscular fitness, that is speed and power.
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
m_ovi_PC = fit(MixedModel, f_ovi, dat, contrasts=contr4)

# ╔═╡ 7eb0443f-eb43-4664-aeda-1bb254fdf241
md"""
There is a numerical interaction with a z-value > 2.0 for the first PCA (i.e., `BPT` - `Run_r`).  This interaction would really need to be replicated to be taken seriously. It is probably be due to larger "unfitness" gains in boys that girls (i.e., in `BPT`)  relative to the slightly larger health-related "fitness" gains of girls than boys (i.e., in `Run_r`). 

**We will look at the PCA solutions in some detail in a separate tutorial.**
"""

# ╔═╡ 830b5655-50c0-425c-a8de-02743867b9c9
md"""
## 3. Other topics

### 3.1 Contrasts are re-parameterizations of the same model

The choice of contrast does not affect the model objective, in other words, they  all yield the same goodness of fit. It does not matter whether a contrast is orthogonal or not. 
"""

# ╔═╡ 6492eb08-239f-42cd-899e-9e227999d9e3
[objective(m_ovi_SeqDiff), objective(m_ovi_Helmert), objective(m_ovi_Hypo), objective(m_ovi_PC)]


# ╔═╡ 74d7a701-91cf-4dd1-9991-f94e156c3291
md"""
### 3.2 VCs and CPs depend on contrast coding

Trivially, the meaning of a contrast depends on its definition. Consequently, the contrast specification has a big effect on the random-effect structure. As an illustration, we refit the LMMs with variance components (VCs) and correlation parameters (CPs) for `Child`-related contrasts of `Test`. Unfortunately, it is not easy, actually rather quite difficult, to grasp the meaning of correlations of contrast-based effects; they represent two-way interactions.
"""

# ╔═╡ 61e9e6c3-31f5-4037-b0f0-40b82d1d8fdc
begin
	f_Child = @formula zScore ~ 1 + Test*a1*Sex + (1 + Test | Child);
	m_Child_SDC = fit(MixedModel, f_Child, dat, contrasts=contr1);
	m_Child_HeC = fit(MixedModel, f_Child, dat, contrasts=contr2);
	m_Child_HyC = fit(MixedModel, f_Child, dat, contrasts=contr3);
	m_Child_PCA = fit(MixedModel, f_Child, dat, contrasts=contr4);
end

# ╔═╡ bbeed889-dc08-429e-9211-92e7be35ec97
VarCorr(m_Child_SDC)

# ╔═╡ f593f70b-ec70-4e96-8246-f1b8203199e0
VarCorr(m_Child_HeC)

# ╔═╡ 2ab82d40-e65a-4801-b118-6cfe340708c0
VarCorr(m_Child_HyC)

# ╔═╡ ba86ee13-a36f-4413-9469-caf168459b5a
VarCorr(m_Child_PCA)

# ╔═╡ 32b02ec9-d3c9-43f9-940e-52f288778d56
md"""
The CPs for the various contrasts are in line with expectations. For the SDC we observe substantial negative CPs between neighboring contrasts. For the orthogonal HeC, all CPs are small; they are uncorrelated. HyC contains some of the SDC contrasts and we observe again the negative CPs. The (roughly) PCA-based contrasts are small with one exception; there is a sizeable CP of +.41 between GM and the core of adjusted physical fitness (c234.15). 

Do these differences in CPs imply that we can move to zcpLMMs when we have orthogonal contrasts? We pursue this question with by refitting the four LMMs with zerocorr() and compare the goodniess of fit.
"""

# ╔═╡ ebeecc4e-1428-4674-9678-68a7d5c04391
begin
	f_Child0 = @formula zScore ~ 1 + Test*a1*Sex + zerocorr(1 + Test | Child);
	m_Child_SDC0 = fit(MixedModel, f_Child0, dat, contrasts=contr1);
	m_Child_HeC0 = fit(MixedModel, f_Child0, dat, contrasts=contr2);
	m_Child_HyC0 = fit(MixedModel, f_Child0, dat, contrasts=contr3);
	m_Child_PCA0 = fit(MixedModel, f_Child0, dat, contrasts=contr4);
end

# ╔═╡ fb5f0942-f148-4bf8-bf01-2a848205b906
MixedModels.likelihoodratiotest(m_Child_SDC0, m_Child_SDC)

# ╔═╡ 8dd002eb-3160-4c5e-b38a-c293c92264ce
MixedModels.likelihoodratiotest(m_Child_HeC0, m_Child_HeC)

# ╔═╡ b74d474f-6bb3-421c-9a2f-32f90f26ee27
MixedModels.likelihoodratiotest(m_Child_HyC0, m_Child_HyC)

# ╔═╡ 12ffa71e-0623-4da0-a576-6268172e4eaf
MixedModels.likelihoodratiotest(m_Child_PCA0, m_Child_PCA)

# ╔═╡ 519fa02a-df7a-4a15-90f1-f951a21f82b6
md"""
Obviously, we can not drop CPs from any of the LMMs. The full LMMs all have the same objective, but we can compare the goodness-of-fit statistics of zcpLMMs more directly. 
"""

# ╔═╡ 23af26f4-8d15-4760-bf3d-51bdc5adb3c0
begin
	zcpLMM=["SDC0", "HeC0", "HyC0", "PCA0"]
	mods = [m_Child_SDC0, m_Child_HeC0, m_Child_HyC0, m_Child_PCA0];
	gof_summary = sort!(DataFrame(zcpLMM=zcpLMM, dof=dof.(mods), 
	deviance=deviance.(mods), AIC = aic.(mods), BIC = bic.(mods)), :deviance)
end

# ╔═╡ 10db1b0f-1389-4beb-98a9-d59822254370
md"""
The best fit was obtained for the PCA-based zcpLMM. Somewhat surprisingly the second best fit was obtained for the SDC. The relatively poor performance of HeC-based zcpLMM is puzzling to me. I thought it might be related to imbalance in design in the present data, but this does not appear to be the case. The same comparison of SeqDiff and Helmert Coding also showed a worse fit for the zcp-HeC LMM than the zcp-SDC LMM. 
"""

# ╔═╡ 65c10c72-cbcf-4923-bdc2-a23aa6ddd856
md"""
### 3.3 VCs and CPs depend on random factor
VCs and CPs resulting from a set of test contrasts can also be estimated for the random factor `School`. Of course, these VCs and CPs may look different from the ones we just estimated for `Child`. 

The effect of `age` (i.e., developmental gain) varies within `School`. Therefore, we also include its VCs and CPs in this model; the school-related VC for `Sex` was not significant. 
"""

# ╔═╡ 9f1d9769-aab3-4e7f-9dc0-6032361d279f
begin
	f_School = @formula zScore ~  1 + Test*a1*Sex + (1 + Test + a1 | School);
	m_School_SeqDiff = fit(MixedModel, f_School, dat, contrasts=contr1);
	m_School_Helmert = fit(MixedModel, f_School, dat, contrasts=contr2);
	m_School_Hypo = fit(MixedModel, f_School, dat, contrasts=contr3);
	m_School_PCA = fit(MixedModel, f_School, dat, contrasts=contr4);
end

# ╔═╡ cbd98fc6-4d7e-42f0-84d0-525ab947187a
VarCorr(m_School_SeqDiff)

# ╔═╡ d8bcb024-a971-47d6-a3f3-9eff6f3a9386
VarCorr(m_School_Helmert)

# ╔═╡ c47058ac-31e6-4b65-8116-e313049f6323
VarCorr(m_School_Hypo)

# ╔═╡ 90bb43c0-3a8c-4958-8a16-2b8886fd3a28
VarCorr(m_School_PCA)

# ╔═╡ aae9f4ae-512a-40b6-8734-6bec9fd3ef99
md" We compare again how much of the fit resides in the CPs."

# ╔═╡ 14122e58-a9dc-420f-8e5e-5572419f0a93
begin
  f_School0 = @formula zScore ~ 1 + Test*a1*Sex + zerocorr(1 + Test + a1 | School);
  m_School_SDC0 = fit(MixedModel, f_School0, dat, contrasts=contr1);
  m_School_HeC0 = fit(MixedModel, f_School0, dat, contrasts=contr2);
  m_School_HyC0 = fit(MixedModel, f_School0, dat, contrasts=contr3);
  m_School_PCA0 = fit(MixedModel, f_School0, dat, contrasts=contr4);
  # 	
  zcpLMM2=["SDC0", "HeC0", "HyC0", "PCA0"]
  mods2 = [m_School_SDC0, m_School_HeC0, m_School_HyC0, m_School_PCA0];
  gof_summary2 = sort!(DataFrame(zcpLMM=zcpLMM2, dof=dof.(mods2), 
  deviance=deviance.(mods2), AIC = aic.(mods2), BIC = bic.(mods2)), :deviance)
end

# ╔═╡ 4f84c162-d2c8-4345-9f9f-5da26f331437
md""" For the random factor `School` the Helmert contrast, followed by PCA-based contrasts have least information in the CPs; SDC is has the largest contribution from CPs. Interesting.
"""

# ╔═╡ e84d2f33-ab76-4cd9-989c-0d27a22907ce
md"""
## 5. That's it
That's it for this tutorial. It is time to try you own contrast coding. You can use these data; there are many alternatives to set up hypotheses for the five tests. Of course and even better, code up some contrasts for data of your own.

Have fun!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Arrow = "~1.6.2"
CSV = "~0.8.5"
DataFrames = "~1.2.2"
MixedModels = "~4.1.1"
RCall = "~0.13.12"
StatsBase = "~0.33.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "DataAPI", "Dates", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "UUIDs"]
git-tree-sha1 = "b00e6eaba895683867728e73af78a00218f0db10"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "1.6.2"

[[ArrowTypes]]
deps = ["UUIDs"]
git-tree-sha1 = "a0633b6d6efabf3f76dacd6eb1b3ec6c42ab0552"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "1.2.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "42ac5e523869a84eac9669eaceed9e4aa0e1587b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.4"

[[BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "f50b5a99aa6ff9db7bf51255b5c21c8bc871ad54"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.5"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "RecipesBase", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "1562002780515d2573a4fb0c3715e4e57481075e"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "d19cd9ae79ef31774151637492291d75194fc5fa"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "c2dbc7e0495c3f956e4615b78d03c7aa10091d0c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.15"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7c365bdef6380b29cfc5caaf99688cd7489f9b87"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.2"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "b3e5984da3c6c95bcf6931760387ff2e64f508f3"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.1"

[[JSONSchema]]
deps = ["HTTP", "JSON", "ZipFile"]
git-tree-sha1 = "b84ab8139afde82c7c65ba2b792fe12e01dd7307"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.3"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "f318e42a48ec0a856292bafeec6b07aed3f6d600"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.1.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "d80cb3327d1aeef0f59eacf225e000f86e4eee0a"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.3"

[[NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2b597c46900f5f811bec31f0dcc88b45744a2a09"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.0+0"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "80a056277142a340e646beea0e213f9aecb99caa"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.12"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3fa15c1f8be168e76d59097f66970adc86bfeb95"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.25"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "8445bf99a36d703a09c601f9a57e2f83000ef2ae"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "6c9040665b2da00d30143261aea22c7427aada1c"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.7"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[WinReg]]
deps = ["Test"]
git-tree-sha1 = "808380e0a0483e134081cc54150be4177959b5f4"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "0.3.1"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
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
# ╠═6492eb08-239f-42cd-899e-9e227999d9e3
# ╟─74d7a701-91cf-4dd1-9991-f94e156c3291
# ╠═61e9e6c3-31f5-4037-b0f0-40b82d1d8fdc
# ╠═bbeed889-dc08-429e-9211-92e7be35ec97
# ╠═f593f70b-ec70-4e96-8246-f1b8203199e0
# ╠═2ab82d40-e65a-4801-b118-6cfe340708c0
# ╠═ba86ee13-a36f-4413-9469-caf168459b5a
# ╟─32b02ec9-d3c9-43f9-940e-52f288778d56
# ╠═ebeecc4e-1428-4674-9678-68a7d5c04391
# ╠═fb5f0942-f148-4bf8-bf01-2a848205b906
# ╠═8dd002eb-3160-4c5e-b38a-c293c92264ce
# ╠═b74d474f-6bb3-421c-9a2f-32f90f26ee27
# ╠═12ffa71e-0623-4da0-a576-6268172e4eaf
# ╟─519fa02a-df7a-4a15-90f1-f951a21f82b6
# ╠═23af26f4-8d15-4760-bf3d-51bdc5adb3c0
# ╟─10db1b0f-1389-4beb-98a9-d59822254370
# ╟─65c10c72-cbcf-4923-bdc2-a23aa6ddd856
# ╠═9f1d9769-aab3-4e7f-9dc0-6032361d279f
# ╠═cbd98fc6-4d7e-42f0-84d0-525ab947187a
# ╠═d8bcb024-a971-47d6-a3f3-9eff6f3a9386
# ╠═c47058ac-31e6-4b65-8116-e313049f6323
# ╠═90bb43c0-3a8c-4958-8a16-2b8886fd3a28
# ╟─aae9f4ae-512a-40b6-8734-6bec9fd3ef99
# ╠═14122e58-a9dc-420f-8e5e-5572419f0a93
# ╟─4f84c162-d2c8-4345-9f9f-5da26f331437
# ╟─e84d2f33-ab76-4cd9-989c-0d27a22907ce
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
