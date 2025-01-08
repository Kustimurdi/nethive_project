import Pkg
print(string("Current working dir: ", pwd(), "\n"))

"""
-------------------------------------------------------------------
1. part - setup of arguments and constants
-------------------------------------------------------------------
"""

include("config.jl")
include("helper.jl")
@show ARGS
parsed_args = parse_args(ARGS, s)
println(parsed_args)

include("definitions.jl")
#include("methods.jl")
include("testing.jl")
#run()
#run2()
run3()
println("wtf")

@info string("DONE!")