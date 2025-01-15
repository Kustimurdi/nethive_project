module Defaults

export DEFAULT_INPUT_SIZE, DEFAULT_OUTPUT_SIZE, DEFAULT_N_BEES, DEFAULT_N_EPOCHS

const DEFAULT_INPUT_SIZE = UInt16[28, 28, 1]
const DEFAULT_OUTPUT_SIZE = UInt16(10)
const DEFAULT_N_BEES = UInt16(3)
const DEFAULT_N_EPOCHS = UInt16(5)

"""
function _initialize_exports()
    for name in names(@__MODULE__, all=true)
        if startswith(string(name), "DEFAULT_")  # Only export constants starting with "DEFAULT_"
            export(Symbol(name))
        end
    end
end
"""

end

