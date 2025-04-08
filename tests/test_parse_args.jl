@testset "Argument Parsing" begin
    settings = create_arg_parse_settings()

    @testset "Default values" begin
        parsed_args = parse_args(settings; argv=String[])  # simulate no user input
        for (key, default) in DEFAULTS
            @test haskey(parsed_args, key)
            @test parsed_args[key] == default
        end
    end

    @testset "Custom arguments" begin
        # Replace these with actual argument flags your code expects
        custom_argv = [
            "--grid_size", "25",
            "--n_bees", "200"
        ]
        parsed_args = parse_args(settings; argv=custom_argv)

        @test parsed_args["grid_size"] == 25
        @test parsed_args["n_bees"] == 200
    end
end
