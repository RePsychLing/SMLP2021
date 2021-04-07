### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 4b4967d8-d6fe-403b-9fed-c7f95d9d6920
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
end

# ╔═╡ f5e09c62-9762-411c-ab71-255083c173a1
begin
	using CSV, RCall, DataFrames, DataFramesMeta, MixedModels, JellyMe4
	using CategoricalArrays: levels
end

# ╔═╡ 62ab7f8a-119c-4f2d-abcf-86e5842a6fa8
using InteractiveUtils

# ╔═╡ 07197c6c-9628-11eb-3be8-092b8787a84e
md"""
# Mixed Models Tutorial: Physical fitness in third grade

Ths script uses data reported in Fühner, Golle, Granacher, & Kliegl (2021). 
Physical fitness in third grade of primary school: 
A mixed model analysis of 108,295 children and 515 schools.

It contains if not at all so definitely most of the commands available in the MixedModels.jl 
pacakge for the analysis of _linear_ mixed-effect models.

The models analyzed here are using simpler random-effect structures than those in the 
publication to speed up LMM estimations; they are still quite complex and not or not easily 
handled by other software. 

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

## Read and transform

+ read data
+ center age
+ compute indicator for quadratic trend of age
+ specify contrasts for test factor
"""

# ╔═╡ dce799c0-38cf-46ae-ab36-1e620989dcea
dat = @linq rcopy(R"readRDS('./data/fggk21.rds')") |>
	transform(
       a1 =  :age .- 8.5,
       a2 = (:age .- 8.5) .^2,
       );

# ╔═╡ 37539bd0-c9ad-4946-b024-9df1d77373ef
describe(dat)

# ╔═╡ f3ad7942-a561-4570-97c6-05b76e9017d1
contr = merge(
       Dict(nm => SeqDiffCoding() for nm in (:Test, :Sex)),
       Dict(nm => Grouping() for nm in (:School, :Child, :Cohort)),
   );

# ╔═╡ 55f5f753-d335-4870-a11c-3279bacc61b3
md"""
## Hierarchical tests of VC and CP for Child and School

Two sequences of tests to check that VCs and CPs increase goodness of fit for random facters `Child` and `School`.

### LMM - only varying intercepts
"""

# ╔═╡ b2c182fd-ca35-47a4-a89f-10c7574ad26f
f_ovi = @formula zScore ~ 1 + Test*a1*Sex + 
        (1 | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ 15163958-681f-4a60-b059-84624063f882
m_ovi = fit(MixedModel, f_ovi, dat, contrasts=contr)

# ╔═╡ 37abc84c-fff1-42ed-8f37-85231cf70927
md"""
### Focus on Child

+ add VC to oviLMM
+ add CP to VC
+ compare with reference LMM
"""

# ╔═╡ 045f6831-7a7e-486d-bb7a-3f1aa45cfdcd
f_VC_C = @formula zScore ~ 1 + Test*a1*Sex + 
         (1 | School) + zerocorr(1 + Test | Child) + (1 | Cohort);

# ╔═╡ 5ca91c64-5311-4ccc-bd33-3310e9175d3d
m_VC_C = fit(MixedModel, f_VC_C, dat, contrasts=contr)

# ╔═╡ 6da0675d-48b8-4c7f-a243-457bdf82eb59
f_CP_C = @formula zScore ~ 1 + Test*a1*Sex + 
         (1 | School) + (1 + Test | Child) + (1 | Cohort);

# ╔═╡ 5e1b3620-d0b6-4105-8a06-22bb3d74e425
m_CP_C = fit(MixedModel, f_CP_C, dat, contrasts=contr)

# ╔═╡ 15cb2be4-f990-4c0b-a43f-c0494a87bb9d
md"""
### Compare goodness of fit for Child sequence
"""

# ╔═╡ e6ba5c01-f6d5-4733-89d2-d1c81e505dab
MixedModels.likelihoodratiotest(m_ovi, m_VC_C, m_CP_C)

# ╔═╡ 90c7639c-9807-4655-be6a-d6ee46c47bbd
mods = [m_ovi, m_VC_C, m_CP_C];

# ╔═╡ 614b4261-b4de-45d5-9188-d7e7b623e562
gof_summary = DataFrame(dof=dof.(mods), deviance=deviance.(mods),
              AIC = aic.(mods), AICc = aicc.(mods), BIC = bic.(mods))

# ╔═╡ 75605d98-3a07-482d-9a4c-57a275979475
md"""
Both VCs and CPs significantly improve the goodness of fit.

### Focus on School

+ add VC to oviLMM
+ add CP to VC
+ compare with reference LMM
"""

# ╔═╡ 124bf317-f541-46d3-9aa6-e654d22b6ed7
f_VC_S = @formula zScore ~ 1 + Test*a1*Sex + 
         zerocorr(1 + Test | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ 494df60f-1d3e-4d4b-a658-ec83f831fe23
m_VC_S = fit(MixedModel, f_VC_S, dat, contrasts=contr)

# ╔═╡ 286b3890-7261-4540-8db3-7efe0322b411
f_CP_S = @formula zScore ~ 1 + Test*a1*Sex + 
         (1 + Test | School) + (1 | Child) + (1 | Cohort);

# ╔═╡ c6ee4d76-6608-4eba-871c-68c63d63e059
m_CP_S = fit(MixedModel, f_CP_S, dat, contrasts=contr)

# ╔═╡ 31e4f4eb-05dc-4cc1-91d5-e7fee1f65012
md"### Compare goodness of fit for School sequence"

# ╔═╡ e5c2561b-8110-405c-af77-2603f3c58ed4
MixedModels.likelihoodratiotest(m_ovi, m_VC_S, m_CP_S)

# ╔═╡ 60cc13e4-3495-43f7-8a14-a531c03e39f4
mods_S = [m_ovi, m_VC_S, m_CP_S];

# ╔═╡ 0a41a7f9-bbfe-4612-a40f-dcc61d6d69f1
gof_summary_S = DataFrame(dof=dof.(mods_S), deviance=deviance.(mods_S),
              AIC = aic.(mods_S), AICc = aicc.(mods_S), BIC = bic.(mods_S))

# ╔═╡ 98174664-ce99-487f-930d-9304c0510863
md"""
Both VCs and CPs significantly improve the goodness of fit.

## LMM
"""

# ╔═╡ e52d2680-9bb5-41e4-a39f-9271c575fbeb
f1 = @formula zScore ~ 1 + Test*a1*Sex +
     (1+Test+a1+Sex | School) +  (1+Test | Child) + (1 | Cohort);

# ╔═╡ 483bfe90-d570-43f7-8590-8f48ac6b8bb4
m1 = fit(MixedModel, f1, dat, contrasts=contr)

# ╔═╡ 7c0ef836-d570-47e6-9f1d-b48327381fc2
md"## Fit statistics"

# ╔═╡ 40dde025-1e61-47a4-ade1-cbbec7ca73e0
# MixedModels.OptSummary: get all info with: m1.optsum)

loglikelihood(m1)  # StatsBase.loglikelihood: return loglikelihood of the model

# ╔═╡ 734f8fe3-d2ed-4219-82c0-515e00ef08ee
deviance(m1)   # StatsBase.deviance: negative twice the log-likelihood relative to saturated model

# ╔═╡ 1b13c2df-c5a9-42c1-b72c-4d066b40933a
objective(m1)  # MixedModels.objective: saturated model not clear: negative twice the log-likelihood

# ╔═╡ f7e372ee-7058-4519-b165-ee033a4c4565
# all from StatsBase
nobs(m1) # n of observations; they are not independent

# ╔═╡ 2f14e5bb-b5d1-4caf-a21d-4c15a7ab1388
dof(m1)  # n of degrees of freedom is number of model parameters

# ╔═╡ 3f934f0a-3e05-447a-84b6-466c314281ce
aic(m1)  # objective(m1) + 2*dof(m1)

# ╔═╡ 96f9cc2d-a016-4f8e-8cbc-c94b1386729d
bic(m1)  # objective(m1) + dof(m1)*log(nobs(m1))

# ╔═╡ 11adf098-f022-4207-b3e7-fe62d5766a9a
md"## Fixed effects"

# ╔═╡ 0ff6021c-e15d-40ba-b234-dc5143622456
coeftable(m1)     # StatsBase.coeftable: fixed-effects statiscs; default level=0.95

# ╔═╡ 1b822e97-9ef3-4b4d-837c-384e8f8050ba
CSV.write("./data/m1_fe.csv", DataFrame(coeftable(m1)));

# ╔═╡ 27d94ddc-3bef-4866-b5b6-1b1139654bb6
# ... or parts of the table
coef(m1)              # StatsBase.coef

# ╔═╡ 1224c0cb-9e0f-4e7a-8a88-eddfa808c9bd
fixef(m1)    # MixedModels.fixef: not the same as coef() for rank-deficient case

# ╔═╡ 8f419441-e7e2-4aa9-8001-da4621bc40c8
m1.β                  # alternative extractor

# ╔═╡ a1515e14-20b5-4cd7-9563-c4599fa35083
fixefnames(m1)        # works also for coefnames(m1)

# ╔═╡ 42f609a2-aef2-4a01-a9aa-d5105892a36c
vcov(m1)   # StatsBase.vcov: var-cov matrix of fixed-effects coefficients

# ╔═╡ f80b483d-ee15-4fff-92e4-a0a33bd5137f
vcov(m1; corr=true) # StatsBase.vcov: correlation matrix of fixed-effects coefficients

# ╔═╡ 9f4f2040-ca2d-42a1-ab1a-809f05d959b6
stderror(m1)          # StatsBase.stderror: SE for fixed-effects coefficients

# ╔═╡ ac9b5a2a-2a49-405e-a486-5be7e577aaf4
propertynames(m1)  # names of available extractors

# ╔═╡ ec3ae437-e8b5-481d-b403-e739c8dfb36e
md"""
## Parameter estimates of random-effect structure 

Aka: covariance parameter estimates
"""

# ╔═╡ 7065a0c8-e0d9-4d67-8467-a4bf57ae283a
BlockDescription(m1) #  Description of blocks of A and L in a LinearMixedModel

# ╔═╡ 4d98627b-77d7-400e-bd59-3f23ef4fef35
VarCorr(m1) # MixedModels.VarCorr: estimates of random-effect structure (RES)

# ╔═╡ 646d3b14-45b8-4a2a-93f1-61685108eaef
# ... or parts of the table
#propertynames(m1)
m1.σ     # residual; m1.sigma, MixedModels.sdest(m1), sqrt(MixedModels.varest(m1))

# ╔═╡ 469efe6d-5018-4ad5-bc5a-ac191cba9be6
m1.σs    # VCs; m1.sigmas

# ╔═╡ b44b006f-7556-42a8-8bcb-096b575fa305
m1.θ     # Parameter vector for RES (w/o residual); m1.theta

# ╔═╡ 2bd21aee-e488-4c09-b4de-efc24ceec9c9
md"""
*DB: There's something interesting here.  The s.d. of the intercept for child is almost exactly the same as the residual s.d.*
"""

# ╔═╡ ad27c2df-3276-4e38-a667-dc83460bd845
md"""
*RK: I had not noticed this before, but individual differences in children's physical fitness are much larger than any other effect I have seen for these data.*
"""

# ╔═╡ cea0b827-13ec-4db3-877c-47ffe3c5e2ac
## check singularity
issingular(m1) # Test if model is singular for paramter vector m1.theta (default)

# ╔═╡ 77d6e8c6-173c-4a61-ad23-61620796246b
md"## Effects PCA w/ CPs"

# ╔═╡ d06708d6-357a-48af-8ab6-5c4d4cd3483d
m1_pca=MixedModels.PCA(m1, corr=true);

# ╔═╡ ef4059a9-8755-4ab4-b3ff-923fbad3a600
m1_pca.Child

# ╔═╡ 6c4f7b49-fa7a-481f-bc0d-60b959904f30
m1_pca.School

# ╔═╡ 3e8cb766-ba13-4bff-99e5-17b4517e2d08
md"""
# LMM with CPs for scores of Test (Table 3)

+ The scores LMM `m1` estimates CPs between scores of tests. 
+ This is a reparameterized version of the reference LMM `m1`.
"""

# ╔═╡ 0696b522-cf74-49dc-b4bd-57092cf231a2
f1L = @formula zScore ~ 1 + Test*a1*Sex +
      (0+Test+a1+Sex | School) +  (0+Test | Child) + (1 | Cohort);

# ╔═╡ 39b4e0b9-63d0-4920-8442-8646b0da665c
m1L = fit(MixedModel, f1L, dat, contrasts=contr)

# ╔═╡ cd7f295a-e009-42da-bc72-1581fef855a0
m1L_pca=MixedModels.PCA(m1L, corr=true)

# ╔═╡ 32b773ac-4790-4d42-9753-d00f6f5fc7c0
md"""
- Seems to indicate that a1 term in School random effects can be removed
- Exposes a deficiency in the layout of the results in that `Test: Run` occurs in the random effects standard deviations but not in the fixed effects, so it is not in the table.
- Perhaps use `0 + ` in the fixed-effects too
"""

# ╔═╡ fb2817e2-6b2e-4093-8d60-47eff6b92a5e
md"""
## Compare objective of m1 and m1L
 
m1L is a reparameterization of m1, so should be the same - in practice, should be close.
"""

# ╔═╡ 63cd80d5-4204-4f4f-afc6-05808f9bd0ff
(objective(m1), objective(m1L))

# ╔═╡ f21e0860-1aff-4e70-a58d-fd25be5e6026
md"""
## Conditional modes of random effects

### Extraction
"""

# ╔═╡ 01873fcb-bca5-4403-96dc-ccf80a281666
begin
	cmL = raneftables(m1L);          # better: NamedTuple of columntables

	ChildL  = DataFrame(cmL.Child);  # also: Child_v2 = DataFrame(first(raneftables(m1))); 
	CSV.write("./data/fggk21_ChildL.csv", ChildL);

	SchoolL = DataFrame(cmL.School); # but no extractor for columntables between first and last;
	CSV.write("./data/fggk21_SchoolL.csv", SchoolL);
end

# ╔═╡ d75528a3-d9e3-4143-ac8d-f15f67b9a357
md"- *Perhaps save these as Arrow files instead of CSV?*"

# ╔═╡ cf08b244-5846-4b0a-9320-d32f8c022e03
md"""
### Caterpillar plots


### Borrowing strengths (shrinkage)


# Various control LMMs a

## Age x Sex interactions nested in levels of Test

+ The nested LMM `m1` estimates the age x sex interaction for each (level of) test. 
+ This is a reparameterized version of the reference LMM `m1`.
"""

# ╔═╡ 00a13746-0e2b-4049-a723-420970bd6f04
f1_nested = @formula zScore ~ 0 + Test & (a1*Sex) +
            (0+Test+a1+Sex | School) +  (0+Test | Child) + (1 | Cohort);

# ╔═╡ 168625b8-ba19-4214-ac91-89e7e066a24f
m1_nested = fit(MixedModel, f1_nested, dat, contrasts=contr)

# ╔═╡ 03cf23fc-6fdb-45ad-be38-dc61dca27d6e
# compare objective - m1_nested is a reparameterization of m1
(m1.objective, m1_nested.objective)

# ╔═╡ 3bb9bf27-80f4-4e09-8f21-49c3bea4c28f
md"""
None of the five interaction terms is significant.

## Quadratic trends of age

Check whether adding quadratic trends of age to the reference LMM increases the goodness of fit.
"""

# ╔═╡ a91cb555-3c8d-45af-9fc7-0b1708c21aa5
f1_agesq = @formula zScore ~ 1 + Test*(a1+a2)*Sex  +
           (1+Test+a1+Sex | School) +  (1+Test | Child) + (1 | Cohort);

# ╔═╡ 22fa298b-e42e-4429-b1d0-f36703909b2a
m1_agesq = fit(MixedModel, f1_agesq, dat, contrasts=contr)

# ╔═╡ 67c05bfc-7690-46d7-b9d3-43162f5bfe98
MixedModels.likelihoodratiotest(m1, m1_agesq)

# ╔═╡ abf1b03d-f282-4065-8fe4-86d4c9ac8461
mods3 = [m1, m1_agesq];

# ╔═╡ d0070d36-fbb8-4684-ab18-aecb19a8ffa0
DataFrame(dof=dof.(mods3), deviance=deviance.(mods3), AIC=aic.(mods3),
	AICc=aicc.(mods3), BIC=bic.(mods3))

# ╔═╡ df41f644-2ba0-4e4f-bd4d-ce68d7daa624
md"""
Adding quadratic trends of age to the fixed effects does not increase the goodness of fit.

# Observation level residuals for reference LMM m1

# Transfer (some) model objects to RCall
"""

# ╔═╡ dc4c851a-d2a8-457c-94b2-4bfce4244f3f
begin
	R"require('lme4')"

	m1_j = Tuple([m1, dat]);
	@rput m1_j;


	R"save(m1_j, file='./fits/m1_j.rda')"
end;

# ╔═╡ 8e6419a5-879e-4522-b368-b627b3e4be06
md"""
# Temporary storage of model object

Julia’s built-in serialize/deserialize is not guaranteed to work across 
versions of packages or Julia. Use for saving and restoring data add 
model objects between sessions or for sending them to parallel workers.

# Julia and Package Versions

Not sure how to make this work.  At present the output goes to the process where `Pluto.run()` was executed.
"""

# ╔═╡ b2e8b812-918d-46fa-90f8-f4cc3246c640
begin
	Pkg.status()
end

# ╔═╡ 8855c4fe-f3f3-4f2b-bae8-ca7ff2d4bfe0
versioninfo()

# ╔═╡ Cell order:
# ╟─07197c6c-9628-11eb-3be8-092b8787a84e
# ╠═4b4967d8-d6fe-403b-9fed-c7f95d9d6920
# ╠═f5e09c62-9762-411c-ab71-255083c173a1
# ╠═dce799c0-38cf-46ae-ab36-1e620989dcea
# ╠═37539bd0-c9ad-4946-b024-9df1d77373ef
# ╠═f3ad7942-a561-4570-97c6-05b76e9017d1
# ╟─55f5f753-d335-4870-a11c-3279bacc61b3
# ╠═b2c182fd-ca35-47a4-a89f-10c7574ad26f
# ╠═15163958-681f-4a60-b059-84624063f882
# ╟─37abc84c-fff1-42ed-8f37-85231cf70927
# ╠═045f6831-7a7e-486d-bb7a-3f1aa45cfdcd
# ╠═5ca91c64-5311-4ccc-bd33-3310e9175d3d
# ╠═6da0675d-48b8-4c7f-a243-457bdf82eb59
# ╠═5e1b3620-d0b6-4105-8a06-22bb3d74e425
# ╟─15cb2be4-f990-4c0b-a43f-c0494a87bb9d
# ╠═e6ba5c01-f6d5-4733-89d2-d1c81e505dab
# ╠═90c7639c-9807-4655-be6a-d6ee46c47bbd
# ╠═614b4261-b4de-45d5-9188-d7e7b623e562
# ╟─75605d98-3a07-482d-9a4c-57a275979475
# ╠═124bf317-f541-46d3-9aa6-e654d22b6ed7
# ╠═494df60f-1d3e-4d4b-a658-ec83f831fe23
# ╠═286b3890-7261-4540-8db3-7efe0322b411
# ╠═c6ee4d76-6608-4eba-871c-68c63d63e059
# ╟─31e4f4eb-05dc-4cc1-91d5-e7fee1f65012
# ╠═e5c2561b-8110-405c-af77-2603f3c58ed4
# ╠═60cc13e4-3495-43f7-8a14-a531c03e39f4
# ╠═0a41a7f9-bbfe-4612-a40f-dcc61d6d69f1
# ╟─98174664-ce99-487f-930d-9304c0510863
# ╠═e52d2680-9bb5-41e4-a39f-9271c575fbeb
# ╠═483bfe90-d570-43f7-8590-8f48ac6b8bb4
# ╟─7c0ef836-d570-47e6-9f1d-b48327381fc2
# ╠═40dde025-1e61-47a4-ade1-cbbec7ca73e0
# ╠═734f8fe3-d2ed-4219-82c0-515e00ef08ee
# ╠═1b13c2df-c5a9-42c1-b72c-4d066b40933a
# ╠═f7e372ee-7058-4519-b165-ee033a4c4565
# ╠═2f14e5bb-b5d1-4caf-a21d-4c15a7ab1388
# ╠═3f934f0a-3e05-447a-84b6-466c314281ce
# ╠═96f9cc2d-a016-4f8e-8cbc-c94b1386729d
# ╟─11adf098-f022-4207-b3e7-fe62d5766a9a
# ╠═0ff6021c-e15d-40ba-b234-dc5143622456
# ╠═1b822e97-9ef3-4b4d-837c-384e8f8050ba
# ╠═27d94ddc-3bef-4866-b5b6-1b1139654bb6
# ╠═1224c0cb-9e0f-4e7a-8a88-eddfa808c9bd
# ╠═8f419441-e7e2-4aa9-8001-da4621bc40c8
# ╠═a1515e14-20b5-4cd7-9563-c4599fa35083
# ╠═42f609a2-aef2-4a01-a9aa-d5105892a36c
# ╠═f80b483d-ee15-4fff-92e4-a0a33bd5137f
# ╠═9f4f2040-ca2d-42a1-ab1a-809f05d959b6
# ╠═ac9b5a2a-2a49-405e-a486-5be7e577aaf4
# ╟─ec3ae437-e8b5-481d-b403-e739c8dfb36e
# ╠═7065a0c8-e0d9-4d67-8467-a4bf57ae283a
# ╠═4d98627b-77d7-400e-bd59-3f23ef4fef35
# ╠═646d3b14-45b8-4a2a-93f1-61685108eaef
# ╠═469efe6d-5018-4ad5-bc5a-ac191cba9be6
# ╠═b44b006f-7556-42a8-8bcb-096b575fa305
# ╠═2bd21aee-e488-4c09-b4de-efc24ceec9c9
# ╠═ad27c2df-3276-4e38-a667-dc83460bd845
# ╠═cea0b827-13ec-4db3-877c-47ffe3c5e2ac
# ╟─77d6e8c6-173c-4a61-ad23-61620796246b
# ╠═d06708d6-357a-48af-8ab6-5c4d4cd3483d
# ╠═ef4059a9-8755-4ab4-b3ff-923fbad3a600
# ╠═6c4f7b49-fa7a-481f-bc0d-60b959904f30
# ╟─3e8cb766-ba13-4bff-99e5-17b4517e2d08
# ╠═0696b522-cf74-49dc-b4bd-57092cf231a2
# ╠═39b4e0b9-63d0-4920-8442-8646b0da665c
# ╠═cd7f295a-e009-42da-bc72-1581fef855a0
# ╟─32b773ac-4790-4d42-9753-d00f6f5fc7c0
# ╟─fb2817e2-6b2e-4093-8d60-47eff6b92a5e
# ╠═63cd80d5-4204-4f4f-afc6-05808f9bd0ff
# ╟─f21e0860-1aff-4e70-a58d-fd25be5e6026
# ╠═01873fcb-bca5-4403-96dc-ccf80a281666
# ╟─d75528a3-d9e3-4143-ac8d-f15f67b9a357
# ╟─cf08b244-5846-4b0a-9320-d32f8c022e03
# ╠═00a13746-0e2b-4049-a723-420970bd6f04
# ╠═168625b8-ba19-4214-ac91-89e7e066a24f
# ╠═03cf23fc-6fdb-45ad-be38-dc61dca27d6e
# ╟─3bb9bf27-80f4-4e09-8f21-49c3bea4c28f
# ╠═a91cb555-3c8d-45af-9fc7-0b1708c21aa5
# ╠═22fa298b-e42e-4429-b1d0-f36703909b2a
# ╠═67c05bfc-7690-46d7-b9d3-43162f5bfe98
# ╠═abf1b03d-f282-4065-8fe4-86d4c9ac8461
# ╠═d0070d36-fbb8-4684-ab18-aecb19a8ffa0
# ╟─df41f644-2ba0-4e4f-bd4d-ce68d7daa624
# ╠═dc4c851a-d2a8-457c-94b2-4bfce4244f3f
# ╟─8e6419a5-879e-4522-b368-b627b3e4be06
# ╠═b2e8b812-918d-46fa-90f8-f4cc3246c640
# ╠═62ab7f8a-119c-4f2d-abcf-86e5842a6fa8
# ╠═8855c4fe-f3f3-4f2b-bae8-ca7ff2d4bfe0
