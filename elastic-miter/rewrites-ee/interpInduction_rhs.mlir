module {
  handshake.func @interpInduction_rhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["short_in", "long_in"], resNames = ["B_out"]} {
    %c = ee__repeating_init %long {handshake.name = "ee__repeating_init", initToken = 1 : ui1} : <i1>
    %c_buf = buffer %c, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer", debugCounter = false} : <i1>
    %b = ee__interpolator %short, %c_buf {handshake.name = "interpolate"} : <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
