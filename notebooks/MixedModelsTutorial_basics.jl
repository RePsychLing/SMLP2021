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

# ╔═╡ 880ea5ac-a851-446e-8da9-d5f9f161a932
begin
	using CSV, RCall, DataFrames, MixedModels, Statistics, StatsBase
	using CategoricalArrays, Arrow
	using AlgebraOfGraphics
	using AlgebraOfGraphics: linear
	using CairoMakie  # for Scatter
end

# ╔═╡ eebef932-1d63-41e5-998f-9748379c43af
md"""
# Mixed Models Tutorial: Basics

Ths script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

To circumvent delays associated with model fitting we work with models that are less complex than those in the reference publication. All the data to reproduce the models in the publication are used here, too; the script requires only a few changes to specify the more complex models in the paper. 

The script is structured in three main sections: 

1. The **Setup** with reading and examing the data, plotting the main results, and specifying the contrasts for the fixed factor `Test`.
2. A demonstration of **Model complexification** to determine a random-effect structure appropriate for and supported by the data.
3. A **Glossary of MixedModels.jl commands** to inspect the information generated for a fitted model object. 

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

# ╔═╡ a2603e75-9d53-48ea-bd2c-6b2fca03fe55
md"""
#### Checks
"""

# ╔═╡ 6cd5010f-fb4f-4647-a17a-ae4262f03b17
dat_a1 = describe(dat)

# ╔═╡ 16c3cdaa-49fa-46d5-a844-03094329fe4c
viewdf(dat)

# ╔═╡ bbb8b977-29a6-4b0c-b71d-6d37bf22ce19
# ... by Test
begin
	dat_a2 = combine(groupby(dat, [:Test]), 
                             :score => length, :age => mean,
                             :score => mean, :score  => std, 
                             :zScore => mean, :zScore => std)
	viewdf(dat_a2)
end

# ╔═╡ 64cc1f8e-f831-4a53-976f-dc7600b5634d
# ... by Test and Sex
begin
	dat_a3 = combine(groupby(dat, [:Test, :Sex]), 
                             :score => length, :age => mean,
                             :score => mean, :score  => std, 
                             :zScore => mean, :zScore => std)
	viewdf(dat_a3)
end

# ╔═╡ 57693079-d489-4e8c-a745-338ccde7eab1
md"""
### 1.3 Plot main results
The main results of relevance for this tutorial are shown in this figure. 
There are developmental gains within the ninth year of life for each of
the five tests and the tests differ in the magnitude of these gains. 
Also boys outperform girls on each test and, again, the sex difference 
varies across test.
"""

# ╔═╡ bdaea7b2-406b-4a31-9415-4d956f997c6d
md"""
#### Recode and relevel test names to physical components 
"""

# ╔═╡ fcff6dad-7447-4d7a-886f-faa75836b31f
begin
	recode!(dat.Test, "Run" => "Endurance", "Star_r" => "Coordination",
	  "S20_r" => "Speed", "SLJ" => "PowerLOW", "BPT" => "PowerUP");
	levels!(dat.Sex,  ["Boys", "Girls"]);
end

# ╔═╡ 58d561e2-77c0-43c2-990d-008286de4e5c
md"""#### Compute means"""

# ╔═╡ 2880f9af-230d-494b-ad93-119298bf0339
df = combine(
	groupby(
		select(dat, :, :age => (x -> round.(x, digits=1)) => :Age),
		[:Sex, :Test, :Age],
	),
	:zScore => mean => :zScore,
	:zScore => length => :n
	);

# ╔═╡ fb4ab041-9087-4dde-80b1-a1c3cd42044b
md"""#### Generate figure with AoG"""

# ╔═╡ cc71658a-b688-4caa-b783-9e79ca79b609
begin
	design1 = mapping(:age, :zScore, linetype = :Sex, layout_x = :Test);
	design2 = mapping(:Age, :zScore, color = :Sex, layout_x = :Test);
	lines = design1 * linear;
	means = design2 * visual(Scatter, color= [:blue, :red], markersize=6);
	data(df) * means + data(dat) * lines |> draw
end

# ╔═╡ a41bc417-3ca7-4f14-be88-5a91d236e88f
md"""
_Figure 1._ Performance differences between 8.0 und 9.0 years by sex 
in the five physical fitness tests presented as z-transformed data computed
separately for each test. _Endurance_ = cardiorespiratory endurance (i.e., 6-min-
run test), _Coordination_ = star-run test, _Speed_ = 20-m linear sprint test, 
_PowerLOW_ = power of lower limbs (i.e., standing long jump test), _PowerUP_ = 
apower of upper limbs (i.e., ball push test), SD = standard deviation. Points 
are binned observed child means; lines are simple regression fits to the observations.
"""

# ╔═╡ c6c6f056-b8b9-4190-ac14-b900bafa04df
md"""
### 1.4 _SeqDiffCoding_ of `Test`

_SeqDiffCoding_ was used in the publication. This specification tests pairwise 
differences between the five neighboring levels of `Test`, that is: 

+ H1: `Star_r` - `Run` (2-1)
+ H2: `S20_r` - `Star_r` (3-2) 
+ H3: `SLJ` - `S20_r` (4-3)
+ H4: `BPT` - `SLJ` (5-4)

The levels were sorted such that these contrasts map onto four  _a priori_ hypotheses; 
in other words, they are _theoretically_ motivated pairwise comparisons. 
The motivation also encompasses theoretically motivated interactions with `Sex`. 
The order of levels can also be explicitly specified during contrast construction. 
This is very useful if levels are in a different order in the dataframe.

The statistical disadvantage of _SeqDiffCoding_ is that the contrasts are not orthogonal, 
that is the contrasts are correlated. This is obvious from the fact that levels 2, 3, 
and 4 are all used in two contrasts. One consequence of this is that correlation parameters
estimated between neighboring contrasts (e.g., 2-1 and 3-2) are difficult to interpret. 
Usually, they will be negative because assuming some practical limitations on the overall 
range (e.g., between levels 1 and 3), a small "2-1" effect "correlates" negatively with a 
larger "3-2" effect for mathematical reasons. 

Obviously, the tradeoff between theoretical motivation and statistical purity is something 
that must be considered carefully when planning the analysis. 

Various options for contrast coding are the topic of the *MixedModelsTutorial_contrasts.jl*
notbebook.
"""

# ╔═╡ c5326753-a03b-4739-a82e-90ffa7c1ebdb
begin
	recode!(dat.Test, "Endurance"  => "Run", "Coordination" =>  "Star_r",
	  "Speed" => "S20_r", "PowerLOW" => "SLJ", "PowerUP" => "BPT");
	levels!(dat.Sex,  ["Girls", "Boys"]);
	contr = merge(
        Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
		Dict(:Sex => EffectsCoding(; levels=["Girls", "Boys"])),
	    Dict(:Test => SeqDiffCoding(; levels=["Run", "Star_r", "S20_r", "SLJ", "BPT"])),
        Dict(:TestHC => HelmertCoding(; levels=["S20_r", "SLJ", "Star_r", "Run", "BPT"])),
	   )
end;

# ╔═╡ f7d6782e-dcd3-423c-a7fe-c125d8e4f810
md"""
## 2. Model complexification

We fit and compare three LMMs with the same fixed-effect structure but increasing complexity of the random-effect structure for `School`. We ignore the other two random factors `Child` and `Cohort` to avoid undue delays when fitting the models.

1. LMM `m_ovi`: allowing only varying intercepts ("Grand Means");
2. LMM `m_zcp`: adding variance components (VCs) for the four `Test` contrasts, `Sex`, and `age` to LMM `m_ovi`, yielding the zero-correlation parameters LMM;
3. LMM `m_cpx`: adding correlation parameters (CPs) to LMM `m_zcp`; yielding a complex LMM. 

In a final part illustrate how to check whether the complex model is supported by the data, rather than leading to a singular fit and, if supported by the data, whether there is an increase in goodness of fit associated with the model complexification. 

### 2.1 LMM `m_ovi`

In its random-effect structure (RES) we  only vary intercepts (i.e., Grand Means) for `School` (LMM `m_ovi`), that is we allow that the schools differ in the average fitness of its children, average over the five tests.

It is well known that such a simple RES is likely to be anti-conservative with respect to fixed-effect test statistics. 
"""

# ╔═╡ 9a885065-59db-47a0-aa0a-8574f136a834
begin
	f_ovi = @formula zScore ~ 1 + Test * Sex * a1 + (1 | School);
	m_ovi = fit(MixedModel, f_ovi, dat, contrasts=contr)
end

# ╔═╡ 8a302aae-565e-4a5c-8285-881dd36d873d
md"""
### 2.2 LMM `m_zcp`

In this LMM we allow that schools differ not only in `GM`, but also in the size of the four contrasts defined for `Test`, in the difference between boys and girls (`Sex`) and the developmental gain children achieve within the third grade (`age`). 

We assume that there is covariance associated with these CPs beyond residual noise, that is we assume that there is no detectable evidence in the data that the CPs are different from zero.
"""

# ╔═╡ c5c3172f-9801-49c8-a915-6c78c7469ae6
begin
	f_zcp = @formula zScore ~ 1 + Test * Sex * a1 +
                     zerocorr(1 + Test + Sex + a1 | School);
	m_zcp = fit(MixedModel, f_zcp, dat, contrasts=contr)
end

# ╔═╡ 5a02bce6-cbd4-4bfa-b47e-f59abaea7003
md"""Is the model singular (overparameterized, degenerate)? In other words: Is the model **not** supported by the data?"""


# ╔═╡ 20f1f363-5d4e-45cf-8ed2-dbb97549bc22
md"""
### 2.3 LMM `m_cpx`

In the complex LMM investigated in this sequence we give up the assumption of zero-correlation between VCs. 

"""

# ╔═╡ 2db49116-5045-460b-bb9a-c4544333482e
begin
	f_cpx =  @formula zScore ~ 1 + Test * Sex * a1 +
	                          (1 + Test + Sex + a1| School)
	m_cpx = fit(MixedModel, f_cpx, dat, contrasts=contr)
end

# ╔═╡ 4b0bffc1-8b0f-4594-97f6-5449959bd716
md"""We also need to see the VCs and CPs of the random-effect structure (RES)"""

# ╔═╡ efa892f9-aff9-43e1-922d-d09f9062f172
VarCorr(m_cpx)

# ╔═╡ 565d97ca-99f3-44d5-bca6-49696003a46f
md"""
The  CPs associated with `Sex` are very small. Indeed, they could be removed from the LMM. The VC for `Sex`, however, is needed. The specification of these hierarchical  model comparisons within LMM `m_cpx` is left as an online demonstration.

Is the model singular (overparameterized, degenerate)? In other words: Is the model **not** supported by the data?
"""

# ╔═╡ 9947810d-8d61-4d12-bf3c-37c8451f9779
issingular(m_cpx)

# ╔═╡ f6f8832b-5f62-4762-a8b0-6727e93b21a0
md"""
### 2.4 Model comparisons

The checks of model singularity indicate the all three models are supported by the data. Does model complexification also increase the goodness of fit or are we only fitting noise?

#### 2.4.1 LRT and goodness-of-fit statistics

The models are strictly hierarchically nested. We compare the three LMM with a likelihood-ratio tests (LRT) and AIC and BIC goodness-of-fit statistics.
"""


# ╔═╡ f6b1dd53-718d-4455-80f3-9fbf14e03f54
MixedModels.likelihoodratiotest(m_ovi, m_zcp, m_cpx)

# ╔═╡ 2703966a-b129-49de-81d8-6bc8c5048dbe
begin
	mods = [m_ovi, m_zcp, m_cpx];
	gof_summary = DataFrame(dof=dof.(mods), deviance=deviance.(mods),
              AIC = aic.(mods), AICc = aicc.(mods), BIC = bic.(mods))
end

# ╔═╡ df57033d-3148-4a0a-9055-d4c8ede2a0d1
md"""
Both the additions of VCs and the addition of CPs significantly improved the goodness of fit.

#### 2.4.2 Comparing fixed effects of `m_ovi`, `m_zcp`, and `m_cpx`

We check whether enriching the RES changed the significance of fixed effects in the final model.  
"""


# ╔═╡ cd6623f0-2961-4871-995a-0cc2d75dd320
begin
	m_ovi_fe = DataFrame(coeftable(m_ovi));
	m_zcp_fe = DataFrame(coeftable(m_zcp)); 
	m_cpx_fe = DataFrame(coeftable(m_cpx)); 
    m_all = hcat(m_ovi_fe[:, [1, 2, 4]],
		        leftjoin(m_zcp_fe[:, [1, 2, 4]], m_cpx_fe[:, [1, 2, 4]],
			on = :Name, makeunique=true), makeunique=true);
	rename!(m_all, 
    Dict( "Coef." => "b_ovi", "Coef._2" => "b_zcp", "Coef._1" => "b_cpx",         
              "z" => "z_ovi",     "z_2" => "z_zcp",     "z_1" => "z_cpx"))
	m_all2 = round.( m_all[:, [:b_ovi, :b_zcp, :b_cpx, 
				               :z_ovi, :z_zcp, :z_cpx]], digits=2)
	m_all3 = hcat(m_all.Name,  m_all2)
end 

# ╔═╡ bebde50e-86dd-4399-99ab-cb6542536825
m_all.Name;

# ╔═╡ 0aa76f57-10de-4e84-ae26-858250fb50a7
md"""### 2.5 Fitting and dealing with an overparameterized LMM
The LMMs we compared are all supported by the data. This is not to be taken for granted, however. For example, if the number of units (levels) of a grouping factor is small relative to the number of parameters we are trying to estimate, we may end up with an overparameterized / degenerate RES. 

Here we attempt to fit a full CP matrix for the `Cohort` factor for the reduced set of data
"""

# ╔═╡ 09d2c2bb-9ce1-4288-a8b8-a0f67c6ce65a
begin
	f_cpxCohort =  @formula zScore ~ 1 + Test * a1 * Sex +
	                                (1 + Test + a1 + Sex | Cohort)
	m_cpxCohort = fit(MixedModel, f_cpxCohort, dat, contrasts=contr)
end

# ╔═╡ 57263322-5eb6-4a34-8167-aec88ebf1970
issingular(m_cpxCohort)

# ╔═╡ 227a7734-161c-4689-9ff4-d12a5d3ac362
VarCorr(m_cpxCohort)

# ╔═╡ ca1f28f8-57e8-4e85-aa1e-b184b8dee8f0
md"""How about the **zero-correlation parameter** (zcp) version of this LMM?"""

# ╔═╡ d5efd2c3-4890-4067-b6de-fe2aa4198cdb
begin
	f_zcpCohort =  @formula zScore ~ 1 + Test * a1 * Sex +
	                        zerocorr(1 + Test + a1 + Sex| Cohort)
	m_zcpCohort = fit(MixedModel, f_zcpCohort, dat, contrasts=contr)
end

# ╔═╡ 6654f9af-0235-40db-9ebb-05f8264a4f61
issingular(m_zcpCohort)

# ╔═╡ 45ea65a2-7a32-4da6-9be6-429c20d94283
VarCorr(m_zcpCohort)

# ╔═╡ 43693b8b-0365-4177-8655-c4c00881d185
md"""It looks like the problem is with cohort-related variance in the `Sex` effect. Let's take it out.""" 

# ╔═╡ 9f998859-add7-4c40-bd05-eab6292733c4
begin
	f_zcpCohort_2 =  @formula zScore ~ 1 + Test * a1 * Sex +
	                          zerocorr(1 + Test + a1| Cohort)
	m_zcpCohort_2 = fit(MixedModel, f_zcpCohort_2, dat, contrasts=contr)
end

# ╔═╡ 04584939-9365-4eff-b15f-0f6c5b2b8f89
issingular(m_zcpCohort_2)

# ╔═╡ 4f34f0c8-ec27-4278-944e-1225f853a371
md"""This does solve the problem. Does LMM *m_cpxCohort* fit noise relative to the zcp LMMs?"""

# ╔═╡ d98585a6-0d0d-4d5a-8eec-8c79d80745db
MixedModels.likelihoodratiotest(m_zcpCohort_2, m_zcpCohort, m_cpxCohort)

# ╔═╡ 074ececf-8dff-44cf-85d7-d48118e4e507
md"""I would answer with "mostly yes".  How about extending the reduced LMM again with CPs? This would be a **parsimonious** LMM."""

# ╔═╡ 510e8f45-7440-4ab4-a672-f91085793e91
begin
	f_prmCohort =  @formula zScore ~ 1 + Test * a1 * Sex +  
	 								(1 + Test + a1| Cohort)
	m_prmCohort = fit(MixedModel, f_prmCohort, dat, contrasts=contr)
end

# ╔═╡ c0e0f3f8-fe60-4cc2-88b0-f81ac9154a3c
issingular(m_prmCohort)

# ╔═╡ a824d97b-e235-4411-a7cf-af123bfe6c4f
MixedModels.likelihoodratiotest(m_zcpCohort_2, m_prmCohort, m_cpxCohort)

# ╔═╡ f992cf2c-b0c0-4a96-a51d-289bb6fbe317
begin
	gof_summary2 = DataFrame(dof=dof.([m_zcpCohort_2, m_prmCohort, m_cpxCohort]),
		deviance=deviance.([m_zcpCohort_2, m_prmCohort, m_cpxCohort]),
                AIC = aic.([m_zcpCohort_2, m_prmCohort, m_cpxCohort]), 
		      AICc = aicc.([m_zcpCohort_2, m_prmCohort, m_cpxCohort]), 
		        BIC = bic.([m_zcpCohort_2, m_prmCohort, m_cpxCohort]))
end

# ╔═╡ 3e16b4fa-6d42-4ac4-9295-9ac04b6b855e
VarCorr(m_prmCohort)

# ╔═╡ 47fd8482-81b5-4dd9-ac0a-67861a32e7ed
md"""The decision about model selection will depend on where on the dimension with confirmatory and exploratory poles the analysis is situated."""

# ╔═╡ e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
md"""
#### 2.6 Fitting the published LMM `m1` to the reduced data

The LMM `m1` reported in Fühner et al. (2021) included random factors for `School`,
`Child`, and `Cohort`. The RES for `School` was specified like in LMM `m_cpx`. The
RES for `Child` included VCs an d CPs for `Test`, but not for linear developmental
gain in the ninth year of life `a1` or `Sex`; they are between-`Child` effects. 

The RES for `Cohort` included only VCs, no CPs for `Test`. The _parsimony_ was due
to the small number of nine levels for this grouping factor. We will check online
whether a more complex LMM would be supported by the data. 

Here we fit this LMM `m1` for the reduced data. On a MacBook Pro [13 | 15 | 16] this
takes [303 | 250 | 244 ] s; for LMM `m1a` (i.e., dropping 1 school-relate VC for
`Sex`), times are  [212 | 165 | 160] s. The corresponding `lme4` times for LMM `m1`
are [397  | 348 | 195]. 

Finally, times for fitting the full set of data --not in this script--, for LMM `m1`are 
[60 | 62 | 85] minutes (!); for LMM `m1a` the times were [46 | 48 | 34] minutes. It was 
not possible to fit the full set of data with `lme4`; after about 13 to 18 minutes the 
program stopped with:  `Error in eval_f(x, ...) : Downdated VtV is not positive definite.`
"""



# ╔═╡ 0dd0d060-81b4-4cc0-b306-dda7589adbd2
begin
	f1 =  @formula zScore ~ 1 + Test * a1 * Sex +
	                       (1 + Test + a1 + Sex | School) + (1 + Test | Child) +
					        zerocorr(1 + Test | Cohort)
	m1 = fit(MixedModel, f1, dat, contrasts=contr)
end

# ╔═╡ b8728c1a-b20e-4e88-ba91-927b6272c6d5
issingular(m1)

# ╔═╡ 5f7a0626-92af-4ec7-aa86-e3559efab511
VarCorr(m1)

# ╔═╡ 242d5f20-b42a-4043-9fe1-5b272cf60194
md"""Let's remove the school-related sex VC."""

# ╔═╡ 1b2d8df1-4de9-4546-a39e-116e00d9ef95
begin
	f1a =  @formula zScore ~ 1 + Test * a1 * Sex +
	                        (1 + Test + a1 | School) + (1 + Test | Child) +
				    zerocorr(1 + Test | Cohort)
	m1a = fit(MixedModel, f1a, dat, contrasts=contr)
end

# ╔═╡ a59cb90a-9d16-41d6-a9a6-35b38a08893e
issingular(m1a)

# ╔═╡ 730d919b-30f5-409b-a79f-598e809e368a
VarCorr(m1a)

# ╔═╡ abfd9185-b77b-4b60-99f8-9bbb89914a77
MixedModels.likelihoodratiotest(m1a, m1)

# ╔═╡ e3819941-6254-46a1-97d8-7944ef23f1bc
md"""After removal of the non-significant school-related VC for `Sex` the model is no longer overparameterized for the reduced data set."""

# ╔═╡ 46e4c42d-fc89-4d12-b700-6c3f087e64ca
md"""
## 3. Glossary of _MixedModels.jl_ commands

Here we introduce most of the commands available in the _MixedModels.jl_ 
package that allow the immediated inspection and analysis of results returned in a fitted _linear_ mixed-effect model. 

Postprocessing related to conditional modes will be dealt with in a different tutorial.

### 3.1 Overall summary statistics

```
+ julia> m1.optsum         # MixedModels.OptSummary:  gets all info 
+ julia> loglikelihood(m1) # StatsBase.loglikelihood: return loglikelihood
							 of the model
+ julia> deviance(m1)      # StatsBase.deviance: negative twice the log-likelihood
							 relative to saturated mode`
+ julia> objective(m1)     # MixedModels.objective: saturated model not clear:
							 negative twice the log-likelihood
+ julia> nobs(m1)          # n of observations; they are not independent
+ julia> dof(m1)           # n of degrees of freedom is number of model parameters
+ julia> aic(m1)           # objective(m1) + 2*dof(m1)
+ julia> bic(m1)           # objective(m1) + dof(m1)*log(nobs(m1))
```
"""


# ╔═╡ 0caf1da1-4d06-49b8-b1b4-d24a860c68d8
m1.optsum            # MixedModels.OptSummary:  gets all info

# ╔═╡ 117132af-b6bb-4b11-a92d-3fc34aff3a63
loglikelihood(m_cpx) # StatsBase.loglikelihood: return loglikelihood of the model

# ╔═╡ e707d474-c553-4142-8b40-d6d944cbc58a
deviance(m_cpx)      # StatsBase.deviance: negative twice the log-likelihood relative to saturated mode`

# ╔═╡ ba7c1050-6b29-4dc2-a97b-552bdc259ac0
objective(m_cpx)    # MixedModels.objective: saturated model not clear: negative twice the log-likelihood

# ╔═╡ 80abc5b9-9d9d-4a67-a542-1be2dc1eca0f
nobs(m1) # n of observations; they are not independent

# ╔═╡ d413615b-1ea8-46b2-837c-52a8f147b456
dof(m1)  # n of degrees of freedom is number of model parameters

# ╔═╡ 2a58411c-c975-4477-8e76-0b9b59000e75
aic(m1)  # objective(m1) + 2*dof(m1)

# ╔═╡ 1caa69ec-f1db-4de5-a30a-3d65fd45f5bd
bic(m1)  # objective(m1) + dof(m1)*log(nobs(m1))

# ╔═╡ f300cd09-9abe-4dcf-91e2-dbba7e87b8c1
md"""
### 3.2 Fixed-effect statistics
```
+ julia> coeftable(m1)     # StatsBase.coeftable: fixed-effects statiscs; 
						     default level=0.95
+ julia> Arrow.write("./data/m_cpx_fe.arrow", DataFrame(coeftable(m1)));
+ julia> coef(m1)          # StatsBase.coef - parts of the table
+ julia> fixef(m1)         # MixedModels.fixef: not the same as coef() 
                             for rank-deficient case
+ julia> m1.beta           # alternative extractor
+ julia> fixefnames(m1)    # works also for coefnames(m1)
+ julia> vcov(m1)          # StatsBase.vcov: var-cov matrix of fixed-effects coef.
+ julia> stderror(m1)      # StatsBase.stderror: SE for fixed-effects coefficients
+ julia> propertynames(m1) # names of available extractors
```
"""


# ╔═╡ 0cc6ca34-7699-4a85-a174-6fdf1a985613
coeftable(m1) # StatsBase.coeftable: fixed-effects statiscs; default level=0.95

# ╔═╡ 222380a0-8b36-4920-a70d-0fa480005748
Arrow.write("./data/m_cpx_fe.arrow", DataFrame(coeftable(m1)));

# ╔═╡ 0c998d91-d4cf-4c71-8f4b-38ac21f8169e
coef(m1)              # StatsBase.coef; parts of the table

# ╔═╡ b7c715ed-4d11-4256-8ce6-0793388d9dfc
fixef(m1)    # MixedModels.fixef: not the same as coef() for rank-deficient case

# ╔═╡ d9e6af5c-a302-481b-a8a9-2f48d01884a5
m1.β                  # alternative extractor

# ╔═╡ 56e4cd41-1cf3-416c-856a-9cc939fe51cb
fixefnames(m1)        # works also for coefnames(m1)

# ╔═╡ 75999511-1600-4ce5-8bb7-6dc841bbbd50
vcov(m1)   # StatsBase.vcov: var-cov matrix of fixed-effects coefficients

# ╔═╡ 815c070a-8cc1-40f3-91bb-7c38f904399b
vcov(m1; corr=true) # StatsBase.vcov: correlation matrix of fixed-effects coefficients

# ╔═╡ 4eef686a-6943-45a8-8309-e1b264af34b5
stderror(m1)       # StatsBase.stderror: SE for fixed-effects coefficients

# ╔═╡ 89a1cfec-2b87-4bf9-95e3-7c7312069af8
propertynames(m1)  # names of available extractors

# ╔═╡ 1ee16dfe-97d6-4958-b9c4-cf2812691057
md"""
### 3.3 Covariance parameter estimates
These commands inform us about the model parameters associated with the RES.
```
+ julia> issingular(m1)        # Test singularity for param. vector m1.theta
+ julia> VarCorr(m1)           # MixedModels.VarCorr: est. of RES
+ julia> propertynames(m1)
+ julia> m1.σ                  # residual; or: m1.sigma
+ julia> m1.σs                 # VCs; m1.sigmas
+ julia> m1.θ                  # Parameter vector for RES (w/o residual); m1.theta
+ julia> MixedModels.sdest(m1) #  prsqrt(MixedModels.varest(m1))
+ julia> BlockDescription(m1)  #  Description of blocks of A and L in an LMM
```
"""

# ╔═╡ 7302b067-270f-4d6a-a0c6-fabd46cc72b0
issingular(m1) # Test if model is singular for paramter vector m1.theta (default)

# ╔═╡ acd9a3eb-6eae-4e09-8932-e44f91e725b4
VarCorr(m1) # MixedModels.VarCorr: estimates of random-effect structure (RES)

# ╔═╡ 1c9de228-93a4-4d1a-82b5-48fe22cfe0f1
m1.σs      # VCs; m1.sigmas

# ╔═╡ 363dcab6-f9f7-42e7-9de7-6a9616cb060f
m1.θ       # Parameter vector for RES (w/o residual); m1.theta

# ╔═╡ f0538b16-e32e-4393-81d2-1d0ce271d619
BlockDescription(m1) #  Description of blocks of A and L in a LinearMixedModel

# ╔═╡ 5a62529e-cc10-4e2c-8d75-125951f2dc34
md"""## 4. Generate figure with CairoMakie"""

# ╔═╡ a7e68776-7a42-4b4a-b540-f26b8b57d520
lab = levels(dat.Test)

# ╔═╡ 10963e51-7edc-4aa2-9535-dfdf81586b76


# ╔═╡ aaeaeca7-438b-441f-83ce-be8de76a4d02
tlabels = [     # establish order and labels of tbl.Test
	"Run" => "Endurance",
	"Star_r" => "Coordination",
	"S20_r" => "Speed",
	"SLJ" => "Power Low",
	"BPT" => "Power Up",
]

# ╔═╡ af370caf-a44e-45d2-a547-231dfc00697c
df2 = groupby(   # summary grouped data frame by test, sex, rounded age
	combine(
		groupby(
			select(dat,
				:age => (x -> round.(x, digits=1)) => :Age,
				:Test => (t -> getindex.(Ref(Dict(tlabels)), t)) => :Test,
				:Sex => (s -> ifelse.(s .== "female", "Girls", "Boys")) => :Sex,
				:zScore,
			),
			[:Age, :Sex, :Test]),
		:zScore => mean => :zScore,
		),
	:Test,
)

# ╔═╡ 1f6446cc-8b40-4cec-880c-3318f78a56f8
begin
				# create the figure and panels (axes) within the figure
	fTest = Figure(resolution = (1000, 600))
	faxs =  [Axis(fTest[1, i]) for i in eachindex(tlabels)]
				# iterate over the test labels in the desired order
	for (i, lab) in enumerate(last.(tlabels))
					# create the label in a box at the top
		Box(fTest[1, i, Top()], backgroundcolor = :gray)
    	Label(fTest[1, i, Top()], lab, padding = (5, 5, 5, 5))
					# split the subdataframe by sex to plot the points
		for df in groupby(df2[(Test = lab,)], :Sex)
			scatter!(
				faxs[i],
				df.Age,				
				df.zScore,
				color=ifelse(first(df.Sex) == "Boys", :blue, :red),
				label=first(df.Sex))
		end
	end
	
	axislegend(faxs[1], position = :lt)  # only one legend for point colors
	faxs[3].xlabel = "Age"               # only one axis label

	hideydecorations!.(faxs[2:end], grid = false)  # y labels on leftmost panel only
	linkaxes!(faxs...)                   # use the same axes throughout
	colgap!(fTest.layout, 10)            # tighten the spacing between panels
	
	fTest
end

# ╔═╡ Cell order:
# ╠═9396fcac-b0b6-11eb-3a60-9f2ce25df953
# ╟─eebef932-1d63-41e5-998f-9748379c43af
# ╠═880ea5ac-a851-446e-8da9-d5f9f161a932
# ╠═faafd359-fb38-4f49-9dfd-19cb4f0c5a54
# ╟─ee85abd9-0172-44d3-b03a-c6a780d33c72
# ╠═39443fb0-64ec-4a87-b072-bc8ad6fa9cf4
# ╟─0051ec83-7c30-4d28-9dfa-4c6f5d0259fa
# ╠═7440c439-9d4c-494f-9ac6-18c9fb2fe144
# ╟─a2603e75-9d53-48ea-bd2c-6b2fca03fe55
# ╠═6cd5010f-fb4f-4647-a17a-ae4262f03b17
# ╠═16c3cdaa-49fa-46d5-a844-03094329fe4c
# ╠═bbb8b977-29a6-4b0c-b71d-6d37bf22ce19
# ╠═64cc1f8e-f831-4a53-976f-dc7600b5634d
# ╟─57693079-d489-4e8c-a745-338ccde7eab1
# ╟─bdaea7b2-406b-4a31-9415-4d956f997c6d
# ╠═fcff6dad-7447-4d7a-886f-faa75836b31f
# ╟─58d561e2-77c0-43c2-990d-008286de4e5c
# ╠═2880f9af-230d-494b-ad93-119298bf0339
# ╟─fb4ab041-9087-4dde-80b1-a1c3cd42044b
# ╠═cc71658a-b688-4caa-b783-9e79ca79b609
# ╟─a41bc417-3ca7-4f14-be88-5a91d236e88f
# ╟─c6c6f056-b8b9-4190-ac14-b900bafa04df
# ╟─c5326753-a03b-4739-a82e-90ffa7c1ebdb
# ╟─f7d6782e-dcd3-423c-a7fe-c125d8e4f810
# ╠═9a885065-59db-47a0-aa0a-8574f136a834
# ╟─8a302aae-565e-4a5c-8285-881dd36d873d
# ╠═c5c3172f-9801-49c8-a915-6c78c7469ae6
# ╟─5a02bce6-cbd4-4bfa-b47e-f59abaea7003
# ╟─20f1f363-5d4e-45cf-8ed2-dbb97549bc22
# ╠═2db49116-5045-460b-bb9a-c4544333482e
# ╟─4b0bffc1-8b0f-4594-97f6-5449959bd716
# ╠═efa892f9-aff9-43e1-922d-d09f9062f172
# ╟─565d97ca-99f3-44d5-bca6-49696003a46f
# ╠═9947810d-8d61-4d12-bf3c-37c8451f9779
# ╟─f6f8832b-5f62-4762-a8b0-6727e93b21a0
# ╠═f6b1dd53-718d-4455-80f3-9fbf14e03f54
# ╠═2703966a-b129-49de-81d8-6bc8c5048dbe
# ╟─df57033d-3148-4a0a-9055-d4c8ede2a0d1
# ╠═cd6623f0-2961-4871-995a-0cc2d75dd320
# ╠═bebde50e-86dd-4399-99ab-cb6542536825
# ╟─0aa76f57-10de-4e84-ae26-858250fb50a7
# ╠═09d2c2bb-9ce1-4288-a8b8-a0f67c6ce65a
# ╠═57263322-5eb6-4a34-8167-aec88ebf1970
# ╠═227a7734-161c-4689-9ff4-d12a5d3ac362
# ╟─ca1f28f8-57e8-4e85-aa1e-b184b8dee8f0
# ╠═d5efd2c3-4890-4067-b6de-fe2aa4198cdb
# ╠═6654f9af-0235-40db-9ebb-05f8264a4f61
# ╠═45ea65a2-7a32-4da6-9be6-429c20d94283
# ╟─43693b8b-0365-4177-8655-c4c00881d185
# ╠═9f998859-add7-4c40-bd05-eab6292733c4
# ╠═04584939-9365-4eff-b15f-0f6c5b2b8f89
# ╟─4f34f0c8-ec27-4278-944e-1225f853a371
# ╠═d98585a6-0d0d-4d5a-8eec-8c79d80745db
# ╟─074ececf-8dff-44cf-85d7-d48118e4e507
# ╠═510e8f45-7440-4ab4-a672-f91085793e91
# ╠═c0e0f3f8-fe60-4cc2-88b0-f81ac9154a3c
# ╠═a824d97b-e235-4411-a7cf-af123bfe6c4f
# ╠═f992cf2c-b0c0-4a96-a51d-289bb6fbe317
# ╠═3e16b4fa-6d42-4ac4-9295-9ac04b6b855e
# ╟─47fd8482-81b5-4dd9-ac0a-67861a32e7ed
# ╟─e3cf1ee6-b64d-4356-af23-1dd5c7a0fec6
# ╠═0dd0d060-81b4-4cc0-b306-dda7589adbd2
# ╠═b8728c1a-b20e-4e88-ba91-927b6272c6d5
# ╠═5f7a0626-92af-4ec7-aa86-e3559efab511
# ╟─242d5f20-b42a-4043-9fe1-5b272cf60194
# ╠═1b2d8df1-4de9-4546-a39e-116e00d9ef95
# ╠═a59cb90a-9d16-41d6-a9a6-35b38a08893e
# ╠═730d919b-30f5-409b-a79f-598e809e368a
# ╠═abfd9185-b77b-4b60-99f8-9bbb89914a77
# ╟─e3819941-6254-46a1-97d8-7944ef23f1bc
# ╟─46e4c42d-fc89-4d12-b700-6c3f087e64ca
# ╠═0caf1da1-4d06-49b8-b1b4-d24a860c68d8
# ╠═117132af-b6bb-4b11-a92d-3fc34aff3a63
# ╠═e707d474-c553-4142-8b40-d6d944cbc58a
# ╠═ba7c1050-6b29-4dc2-a97b-552bdc259ac0
# ╠═80abc5b9-9d9d-4a67-a542-1be2dc1eca0f
# ╠═d413615b-1ea8-46b2-837c-52a8f147b456
# ╠═2a58411c-c975-4477-8e76-0b9b59000e75
# ╠═1caa69ec-f1db-4de5-a30a-3d65fd45f5bd
# ╟─f300cd09-9abe-4dcf-91e2-dbba7e87b8c1
# ╠═0cc6ca34-7699-4a85-a174-6fdf1a985613
# ╠═222380a0-8b36-4920-a70d-0fa480005748
# ╠═0c998d91-d4cf-4c71-8f4b-38ac21f8169e
# ╠═b7c715ed-4d11-4256-8ce6-0793388d9dfc
# ╠═d9e6af5c-a302-481b-a8a9-2f48d01884a5
# ╠═56e4cd41-1cf3-416c-856a-9cc939fe51cb
# ╠═75999511-1600-4ce5-8bb7-6dc841bbbd50
# ╠═815c070a-8cc1-40f3-91bb-7c38f904399b
# ╠═4eef686a-6943-45a8-8309-e1b264af34b5
# ╠═89a1cfec-2b87-4bf9-95e3-7c7312069af8
# ╟─1ee16dfe-97d6-4958-b9c4-cf2812691057
# ╠═7302b067-270f-4d6a-a0c6-fabd46cc72b0
# ╠═acd9a3eb-6eae-4e09-8932-e44f91e725b4
# ╠═1c9de228-93a4-4d1a-82b5-48fe22cfe0f1
# ╠═363dcab6-f9f7-42e7-9de7-6a9616cb060f
# ╠═f0538b16-e32e-4393-81d2-1d0ce271d619
# ╟─5a62529e-cc10-4e2c-8d75-125951f2dc34
# ╠═a7e68776-7a42-4b4a-b540-f26b8b57d520
# ╠═10963e51-7edc-4aa2-9535-dfdf81586b76
# ╠═aaeaeca7-438b-441f-83ce-be8de76a4d02
# ╠═af370caf-a44e-45d2-a547-231dfc00697c
# ╠═1f6446cc-8b40-4cec-880c-3318f78a56f8
