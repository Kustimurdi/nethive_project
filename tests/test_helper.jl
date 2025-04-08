using Test
include("../src/core/definitions.jl")  
include("../src/helper.jl")

@testset "Helper Functions" begin
    @testset "Git Info" begin
        branch = get_git_branch()
        commit = get_git_commit()
        @test isa(branch, String)
        @test isa(commit, String)
    end

    @testset "Data Export" begin
        dummy_data = Dict("a" => 1, "b" => 2)
        file_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_data/tests/test_output.csv"
        export_data(file_path, dummy_data, 1, [Int64(1)], "value")
        @test isfile(file_path)
        rm(file_path, force=true)
    end
end
