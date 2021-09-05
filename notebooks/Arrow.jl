### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 710a9d03-759c-49d3-bfb8-2924ce2aabd7
begin
	using Arrow
	using DataFrameMacros
	using DataFrames
end	

# ╔═╡ cedcf04a-0e27-11ec-0382-619c14877fae
md"""
# Notes on saved data files

## The Arrow storage format

The [Arrow storage format](https://arrow.apache.org) provides a language-agnostic storage and memory specification for columnar data tables, which means "something that looks like a data frame in R".  That is, an arrow table is an ordered, named collection of columns, all of the same length.

The columns can be of different types including numeric values, character strings, and factor-like representations - called *DictEncoded*.

An Arrow file can be read or written from R, Python, Julia and many other languages.  Somewhat confusingly in R and Python the name `feather`, which refers to an earlier version of the storage format, is used in some function names like `read_feather`.

## The Emotikon data
The [SMLP2021 repository](https://github.com/RePsychling/SMLP2021) contains a version of the data from Fühner, Golle, Granacher, & Kliegl (2021) in `notebooks/data/fggk21.arrow`.

The `Arrow` package for Julia does not export any function names, which means that the function to read an Arrow file must be called as `Arrow.Table`.
It returns a *column table*, as described in the [Tables package](https://github.com/JuliaData/Tables.jl).
This is like a read-only data frame, which can be easily converted to a full-fledged `DataFrame` if desired.

This arrangement allows for the `Arrow` package not to depend on the `DataFrames` package, which is a heavy-weight dependency, but still easily produce a `DataFrame` if warranted.
"""

# ╔═╡ 43d16ad6-63d4-4f86-9c6d-d5bb8eaacc46
tbl = Arrow.Table("./data/fggk21.arrow")

# ╔═╡ 0f371f73-abb9-4bd8-9e5f-dbb27c41cf0a
df = DataFrame(tbl)

# ╔═╡ cd848da0-8c80-496e-8853-9ae1c7c870db
md"""
## Avoiding needless repetition

One of the principles of relational database design is that information should not be repeated needlessly.
Each row of `df` is determined by a combination of `Child` and `Test`, together producing a `score`, which has been converted to a `zScore`.

The other columns in the table, `Cohort`, `District`, `School`, `BU`, `age`, and `Sex`, are properties of the `Child`.
Furthermore, `District` and `BU` are properties of the `School`.

Storing these values redundantly in the full table takes up space but, more importantly, allows for inconsistency.
As it stands, a given `Child` could be recorded as being in one `Cohort` for the `Run` test and in another `Cohort` for the `S20_r` test and nothing about the table would detect this as being an error.

The approach used in relational databases is to store the information for `score` in one table that contains only `Child`, `Test` and `score`, store the information for the `Child` in another table including `Cohort`, `School`, `age` and `Sex` and, finally, store the information about `School`, which is just `District` and `BU` in a third table.
These tables can then be combined to create the table to be used for analysis by *joining* the different tables together.

The maintainers of the `DataFrames` package have put in a lot of work over the past few years to make joins quite efficient in Julia.
Thus the processing penalty of reassembling the big table from three smaller tables is minimal.

It is important to note that the main advantage of using smaller tables that are joined together to produce the analysis table is the fact that the information in the analysis table is consistent by design.

## Creating the smaller tables
"""

# ╔═╡ bad08fb6-d545-4f2f-a392-79481f2f1fa5
School = unique(select(df, :School, :District, :BU))

# ╔═╡ 9b40930a-d104-40bf-8d77-d2af164559a1
length(unique(School.School))  # should be 515

# ╔═╡ 05c02272-3c9b-40e3-aaa8-545d1ad8fdbf
filesize(Arrow.write("./data/fggk21_School.arrow", School; compress=:zstd))

# ╔═╡ 74b37e99-c442-4a44-94f4-41a95eb4a8bc
Child = unique(select(df, :Child, :School, :Cohort, :Sex, :age))

# ╔═╡ c47c8496-806b-4004-8732-51eafc485e85
length(unique(Child.Child))  # should be 108295

# ╔═╡ f6dfcf8a-03c9-40e4-b9d1-8ec3f1c0b9d5
filesize(Arrow.write("./data/fggk21_Child.arrow", Child; compress=:zstd))

# ╔═╡ 2f6183e6-8419-4d46-a09c-ac1e094a6cae
md"""
We also save the `zScore` column in the `Score` table as we were unable to generate the same values from this data alone.
Apparently the `zScore` column was generated from a larger data set.
"""

# ╔═╡ da0cf31c-bf8a-40c2-8ece-10dd7a05fdbc
filesize(Arrow.write(
		"./data/fggk21_Score.arrow",
		select(df, :Child, :Test, :score, :zScore);
		compress=:zstd,
    )
)

# ╔═╡ 4c47bd9c-27d1-4f2e-acec-7c2e875e6a62
md"""
Now read the Arrow tables in and reassemble the original table.
"""

# ╔═╡ 9184e686-3a0d-433d-8e8a-92244827e886
Score = DataFrame(Arrow.Table("./data/fggk21_Score.arrow"))

# ╔═╡ 46d9125c-60e8-4ecf-8640-e8fc1cd76eff
Child1 = DataFrame(Arrow.Table("./data/fggk21_Child.arrow"))

# ╔═╡ 81c50f02-0464-4568-bd70-aa8477ba9e90
School1 = DataFrame(Arrow.Table("./data/fggk21_School.arrow"))

# ╔═╡ b8e5fd07-5f27-4c63-9921-072579a2d0bd
df1 = disallowmissing!(leftjoin(
		Score, 
		leftjoin(Child1, School1; on=:School);
		on=:Child,
	),
)

# ╔═╡ 62fceea2-7f3c-4f85-8258-105ee30d2650
md"""
!!! note
	The call to `disallowmissing!` is because the join will create columns that allow for missing values but we know that we should not get missing values in the result.  This call will fail if, for some reason, missing values were created.
"""

# ╔═╡ f5fa1b72-03dd-410c-8bd0-781b2f35ac21
nobsChild = combine(groupby(Score, :Child), nrow => :n)

# ╔═╡ 0610652f-a9d7-4bfc-ab45-1f00e641f304
combine(groupby(nobsChild, :n), nrow)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
DataFrameMacros = "75880514-38bc-4a95-a458-c2aea5a3a702"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

[compat]
Arrow = "~1.6.2"
DataFrameMacros = "~0.1.0"
DataFrames = "~1.2.2"
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

[[BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "f50b5a99aa6ff9db7bf51255b5c21c8bc871ad54"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.5"

[[CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

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

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[DataFrameMacros]]
deps = ["DataFrames"]
git-tree-sha1 = "508d57ef7b78551cf69c2837d80af5017ce57217"
uuid = "75880514-38bc-4a95-a458-c2aea5a3a702"
version = "0.1.0"

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

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

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
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

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

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

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

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─cedcf04a-0e27-11ec-0382-619c14877fae
# ╠═710a9d03-759c-49d3-bfb8-2924ce2aabd7
# ╠═43d16ad6-63d4-4f86-9c6d-d5bb8eaacc46
# ╠═0f371f73-abb9-4bd8-9e5f-dbb27c41cf0a
# ╟─cd848da0-8c80-496e-8853-9ae1c7c870db
# ╠═bad08fb6-d545-4f2f-a392-79481f2f1fa5
# ╠═9b40930a-d104-40bf-8d77-d2af164559a1
# ╠═05c02272-3c9b-40e3-aaa8-545d1ad8fdbf
# ╠═74b37e99-c442-4a44-94f4-41a95eb4a8bc
# ╠═c47c8496-806b-4004-8732-51eafc485e85
# ╠═f6dfcf8a-03c9-40e4-b9d1-8ec3f1c0b9d5
# ╟─2f6183e6-8419-4d46-a09c-ac1e094a6cae
# ╠═da0cf31c-bf8a-40c2-8ece-10dd7a05fdbc
# ╟─4c47bd9c-27d1-4f2e-acec-7c2e875e6a62
# ╠═9184e686-3a0d-433d-8e8a-92244827e886
# ╠═46d9125c-60e8-4ecf-8640-e8fc1cd76eff
# ╠═81c50f02-0464-4568-bd70-aa8477ba9e90
# ╠═b8e5fd07-5f27-4c63-9921-072579a2d0bd
# ╟─62fceea2-7f3c-4f85-8258-105ee30d2650
# ╠═f5fa1b72-03dd-410c-8bd0-781b2f35ac21
# ╠═0610652f-a9d7-4bfc-ab45-1f00e641f304
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
