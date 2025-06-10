using Test
include("../src/core/definitions.jl")  
include("../src/helper.jl")

@testset "Helper Functions" begin
    @testset "Git Info" begin
        branch = String(get_git_branch())
        commit = get_git_commit()
        @test isa(branch, String)
        @test isa(commit, AbstractString)
        @info "Type of commit:" typeof(commit)
    end

    @testset "Data Export" begin
        dummy_data = rand(10, 5)  # 10 rows, 5 columns
        file_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/tests/"
        mkpath(file_path)
        export_data(joinpath(file_path, "testoutput.csv"), dummy_data, "value")
        @test isfile(joinpath(file_path, "testoutput.csv"))
        rm(joinpath(file_path, "testoutput.csv"), force=true)
    end

    @testset "Vector Export" begin
        dummy_data = [1.0, 2.0, 3.0]
        file_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/tests/"
        mkpath(file_path)
        vector_to_dataframe(joinpath(file_path, "testoutput.csv"), dummy_data, :value)
        @test isfile(joinpath(file_path, "testoutput.csv"))
        df = CSV.File(joinpath(file_path, "testoutput.csv")) |> DataFrame
        @test df[!, :value] == dummy_data
    end
end