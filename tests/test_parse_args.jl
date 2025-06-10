@testset "Argument Parsing" begin 
    settings = create_arg_parse_settings()
    @testset "Default values" begin
        parsed_args = cparse_args(settings)  # simulate no user input
        for (key, default) in DEFAULTS
            key = lowercase(string(key))
            @test haskey(parsed_args, key)
            @test parsed_args[key] == default
        end
    end

    @testset "Custom arguments" begin
        # Replace these with actual argument flags your code expects
        custom_argv = [
            "--n_epochs", "25",
            "--n_bees", "200"
        ]
        parsed_args = cparse_args(settings, args=custom_argv)

        @test parsed_args["n_epochs"] == 25
        @test parsed_args["n_bees"] == 200
    end
end

