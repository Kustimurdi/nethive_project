import Pkg
print(string("Current working dir: ", pwd(), "\n"))

"""
-------------------------------------------------------------------
1. part - setup of arguments and constants
-------------------------------------------------------------------
"""

include("config/defaults.jl")
include("config/arg_table.jl")
include("config/constants.jl")
include("helper.jl")
@show ARGS


"""
-------------------------------------------------------------------
2. part - setup of the simulation and run 
-------------------------------------------------------------------
"""

include("definitions.jl")
include("methods.jl")
#include("methods.jl")
#run()
#run2()
run3()
println("wtf")

@info string("DONE!")