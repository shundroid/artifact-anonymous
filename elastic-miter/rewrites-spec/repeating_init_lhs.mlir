module {
  handshake.func @repeating_init_lhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %out = ee__repeating_init %in {handshake.name = "ee__repeating_init", initToken = 1 : ui1} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %out : <i1>
  }
}
