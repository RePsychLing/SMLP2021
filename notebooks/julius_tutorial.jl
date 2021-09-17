#= Introduction to Chain and DataFrameMacros packages

This script uses a subset of data reported in Fühner, Golle, Granacher, & Kliegl (2021). Physical fitness in third grade of primary school: A mixed model analysis of 108,295 children and 515 schools.

All children were between 6.0 and 6.99 years at legal keydate (30 September) of school enrollement, that is they were in their ninth year of life in the third grade. To speed up we work with a reduced data set and less complex models than those in the reference publication. This illustrates also drawing a stratified subsample from a large data set.

=#
using DataFrames
using Chain
using DataFrameMacros

using CSV
using Arrow
using Downloads

using StatsBase
using Dates

using MacroTools: prettify

#= Readme for './data/fggk21.rds'

Number of scores: 525126

 1. Cohort: 9 levels; 2011-2019
 2. School: 515 levels 
 3. Child: 108295 levels; all children are between 8.0 and 8.99 years old
 4. Sex: "Girls" (n=55,086), "Boys" (n= 53,209)
 5. age: testdate - middle of month of birthdate 
 6. Test: 5 levels
     + Endurance (`Run`):  6 minute endurance run [m]; to nearest 9m in 9x18m field
     + Coordination (`Star_r`): star coordination run [m/s]; 9x9m field, 4 x diagonal = 50.912 m
     + Speed(`S20_r`): 20-meters sprint [m/s]
     + Muscle power low (`SLJ`): standing long jump [cm] 
     + Muscle power up (`BPT`): 1-kg medicine ball push test [m] 
 7. score - see units
 =#

 # Read the data

data = DataFrame(Arrow.Table("./data/fggk21.arrow"))
describe(data)

#= Extract a stratified subsample

We extract a random sample of 5 children from the Sex (2) x Test (5) cells of the design. Cohort and School are random.

=# 

dat =
	@chain data begin
  	  @transform(:Sex2 = :Sex == "Girls" ? "female" : "male")
   	  @groupby(:Test, :Sex)
   	  combine(x -> x[sample(1:nrow(x), 5), :])
    end

# Three macros:  @transform, @groupby, and @chain -- one at a time

## transform and @transform -- also note the ternary operator for ifelse

### long
transform(dat, :Sex =>
    ByRow(bla -> bla == "female" ? "girl" : "boy") => :Sex2)

transform(dat, :age => (col -> col .+ 1) => :ageplus)

### short - as used above
@transform(dat, :Sex2 = :Sex == "female" ? "girl" : "boy")

@transform(dat, :ageplus = :age + 1)

@transform(dat, @c :age .- mean(:age))  # @c = columnwise


## groupby and @groupby
@groupby(dat, :Age = round(Int, :age))

## @chain  -- 

### reference to a file
url = "https://github.com/RePsychLing/SMLP2021/raw/main/notebooks/data/fggk21.arrow"


### version 1 - traditional Julia style
df1 = DataFrame(Arrow.Table(Downloads.download(url)));
describe(df1)

### version 2 - with Chain
df2 = @chain url begin
    Downloads.download
    Arrow.Table
    DataFrame
   	@groupby(:Test, :Sex)
   	combine(x -> x[sample(1:nrow(x), 2), :])
    @aside CSV.write("test.csv", _)
    end;
df2
describe(df2)


### look behind the scene
@macroexpand(@chain url begin
    Downloads.download
    Arrow.Table
    DataFrame
    end) |> prettify
