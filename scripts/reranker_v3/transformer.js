
const model = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_5_2_8_16_3_4_130 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_120:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_16640:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = ((gidx1*24)+(lidx0*3));
  var alu1 = (alu0+1);
  var val0 = data1_120[alu1];
  var alu2 = (alu0+2);
  var val1 = data1_120[alu2];
  var val2 = data1_120[alu0];
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu3 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 130; Ridx0++) {
    var alu16 = (alu3+bitcast<i32>((bitcast<u32>(Ridx0)<<7u)));
    var val3 = data2_16640[alu16];
    var val4 = data2_16640[(alu16+1)];
    var val5 = data2_16640[(alu16+2)];
    var val6 = data2_16640[(alu16+3)];
    var alu17 = (Ridx0+-120);
    var alu18 = (Ridx0<120);
    var alu19 = select(0.0f,(f32((alu0==Ridx0))),alu18);
    var alu20 = select(1.0f,0.0f,(val2!=alu17));
    var alu21 = select(alu20,0.0f,alu18);
    var alu22 = (alu19+alu21);
    var alu23 = select(0.0f,(f32((alu1==Ridx0))),alu18);
    var alu24 = select(1.0f,0.0f,(val0!=alu17));
    var alu25 = select(alu24,0.0f,alu18);
    var alu26 = (alu23+alu25);
    var alu27 = select(0.0f,(f32((alu2==Ridx0))),alu18);
    var alu28 = select(1.0f,0.0f,(val1!=alu17));
    var alu29 = select(alu28,0.0f,alu18);
    var alu30 = (alu27+alu29);
    acc0[0] = (acc0[0]+(alu22*val3));
    acc0[1] = (acc0[1]+(alu22*val4));
    acc0[2] = (acc0[2]+(alu22*val5));
    acc0[3] = (acc0[3]+(alu22*val6));
    acc0[4] = (acc0[4]+(alu26*val3));
    acc0[5] = (acc0[5]+(alu26*val4));
    acc0[6] = (acc0[6]+(alu26*val5));
    acc0[7] = (acc0[7]+(alu26*val6));
    acc0[8] = (acc0[8]+(alu30*val3));
    acc0[9] = (acc0[9]+(alu30*val4));
    acc0[10] = (acc0[10]+(alu30*val5));
    acc0[11] = (acc0[11]+(alu30*val6));
  }
  var alu44 = (alu3+(gidx1*3072)+(lidx0*384));
  data0_15360[alu44] = acc0[0];
  data0_15360[(alu44+1)] = acc0[1];
  data0_15360[(alu44+2)] = acc0[2];
  data0_15360[(alu44+3)] = acc0[3];
  data0_15360[(alu44+128)] = acc0[4];
  data0_15360[(alu44+129)] = acc0[5];
  data0_15360[(alu44+130)] = acc0[6];
  data0_15360[(alu44+131)] = acc0[7];
  data0_15360[(alu44+256)] = acc0[8];
  data0_15360[(alu44+257)] = acc0[9];
  data0_15360[(alu44+258)] = acc0[10];
  data0_15360[(alu44+259)] = acc0[11];
}`;

const r_5_2_8_16_4_3_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_16384:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = ((gidx1*3072)+(lidx0*384));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var cast0 = bitcast<u32>(Ridx0);
    var alu14 = (alu1+bitcast<i32>((cast0<<2u)));
    var val0 = data1_15360[alu14];
    var alu15 = (alu0+bitcast<i32>((cast0<<9u)));
    var val1 = data2_16384[alu15];
    var val2 = data1_15360[(alu14+1)];
    var val3 = data2_16384[(alu15+128)];
    var val4 = data1_15360[(alu14+2)];
    var val5 = data2_16384[(alu15+256)];
    var val6 = data1_15360[(alu14+3)];
    var val7 = data2_16384[(alu15+384)];
    var val8 = data1_15360[(alu14+128)];
    var val9 = data1_15360[(alu14+129)];
    var val10 = data1_15360[(alu14+130)];
    var val11 = data1_15360[(alu14+131)];
    var val12 = data1_15360[(alu14+256)];
    var val13 = data1_15360[(alu14+257)];
    var val14 = data1_15360[(alu14+258)];
    var val15 = data1_15360[(alu14+259)];
    var val16 = data2_16384[(alu15+1)];
    var val17 = data2_16384[(alu15+129)];
    var val18 = data2_16384[(alu15+257)];
    var val19 = data2_16384[(alu15+385)];
    var val20 = data2_16384[(alu15+2)];
    var val21 = data2_16384[(alu15+130)];
    var val22 = data2_16384[(alu15+258)];
    var val23 = data2_16384[(alu15+386)];
    var val24 = data2_16384[(alu15+3)];
    var val25 = data2_16384[(alu15+131)];
    var val26 = data2_16384[(alu15+259)];
    var val27 = data2_16384[(alu15+387)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val0*val16)+(val2*val17)+(val4*val18)+(val6*val19));
    acc0[4] = (acc0[4]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val19));
    acc0[5] = (acc0[5]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val19));
    acc0[6] = (acc0[6]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[7] = (acc0[7]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[8] = (acc0[8]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[9] = (acc0[9]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[10] = (acc0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[11] = (acc0[11]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
  }
  var val28 = data3_128[alu0];
  var val29 = data3_128[(alu0+1)];
  var val30 = data3_128[(alu0+2)];
  var val31 = data3_128[(alu0+3)];
  var alu29 = (alu0+alu1);
  data0_15360[alu29] = (acc0[0]+val28);
  data0_15360[(alu29+1)] = (acc0[3]+val29);
  data0_15360[(alu29+2)] = (acc0[6]+val30);
  data0_15360[(alu29+3)] = (acc0[9]+val31);
  data0_15360[(alu29+128)] = (acc0[1]+val28);
  data0_15360[(alu29+129)] = (acc0[4]+val29);
  data0_15360[(alu29+130)] = (acc0[7]+val30);
  data0_15360[(alu29+131)] = (acc0[10]+val31);
  data0_15360[(alu29+256)] = (acc0[2]+val28);
  data0_15360[(alu29+257)] = (acc0[5]+val29);
  data0_15360[(alu29+258)] = (acc0[8]+val30);
  data0_15360[(alu29+259)] = (acc0[11]+val31);
}`;

const r_2_5_5_2_8_8_3_3_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_57600:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_15360:array<f32>;
@compute @workgroup_size(2,8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 5 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx2)<<6u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u)));
  var alu1 = ((gidx1*3072)+(lidx1*384)+alu0);
  var val0 = data1_15360[(alu1+1)];
  var val1 = data1_15360[(alu1+2)];
  var val2 = data1_15360[(alu1+3)];
  var val3 = data1_15360[(alu1+4)];
  var val4 = data1_15360[(alu1+5)];
  var val5 = data1_15360[(alu1+6)];
  var val6 = data1_15360[(alu1+7)];
  var val7 = data1_15360[(alu1+8)];
  var val8 = data1_15360[(alu1+9)];
  var val9 = data1_15360[(alu1+10)];
  var val10 = data1_15360[(alu1+11)];
  var val11 = data1_15360[(alu1+12)];
  var val12 = data1_15360[(alu1+13)];
  var val13 = data1_15360[(alu1+14)];
  var val14 = data1_15360[(alu1+15)];
  var val15 = data1_15360[(alu1+16)];
  var val16 = data1_15360[(alu1+17)];
  var val17 = data1_15360[(alu1+18)];
  var val18 = data1_15360[(alu1+19)];
  var val19 = data1_15360[(alu1+20)];
  var val20 = data1_15360[(alu1+21)];
  var val21 = data1_15360[(alu1+22)];
  var val22 = data1_15360[(alu1+23)];
  var val23 = data1_15360[(alu1+24)];
  var val24 = data1_15360[(alu1+25)];
  var val25 = data1_15360[(alu1+26)];
  var val26 = data1_15360[(alu1+27)];
  var val27 = data1_15360[(alu1+28)];
  var val28 = data1_15360[(alu1+29)];
  var val29 = data1_15360[(alu1+30)];
  var val30 = data1_15360[(alu1+31)];
  var val31 = data1_15360[(alu1+128)];
  var val32 = data1_15360[(alu1+129)];
  var val33 = data1_15360[(alu1+130)];
  var val34 = data1_15360[(alu1+131)];
  var val35 = data1_15360[(alu1+132)];
  var val36 = data1_15360[(alu1+133)];
  var val37 = data1_15360[(alu1+134)];
  var val38 = data1_15360[(alu1+135)];
  var val39 = data1_15360[(alu1+136)];
  var val40 = data1_15360[(alu1+137)];
  var val41 = data1_15360[(alu1+138)];
  var val42 = data1_15360[(alu1+139)];
  var val43 = data1_15360[(alu1+140)];
  var val44 = data1_15360[(alu1+141)];
  var val45 = data1_15360[(alu1+142)];
  var val46 = data1_15360[(alu1+143)];
  var val47 = data1_15360[(alu1+144)];
  var val48 = data1_15360[(alu1+145)];
  var val49 = data1_15360[(alu1+146)];
  var val50 = data1_15360[(alu1+147)];
  var val51 = data1_15360[(alu1+148)];
  var val52 = data1_15360[(alu1+149)];
  var val53 = data1_15360[(alu1+150)];
  var val54 = data1_15360[(alu1+151)];
  var val55 = data1_15360[(alu1+152)];
  var val56 = data1_15360[(alu1+153)];
  var val57 = data1_15360[(alu1+154)];
  var val58 = data1_15360[(alu1+155)];
  var val59 = data1_15360[(alu1+156)];
  var val60 = data1_15360[(alu1+157)];
  var val61 = data1_15360[(alu1+158)];
  var val62 = data1_15360[(alu1+159)];
  var val63 = data1_15360[(alu1+256)];
  var val64 = data1_15360[(alu1+257)];
  var val65 = data1_15360[(alu1+258)];
  var val66 = data1_15360[(alu1+259)];
  var val67 = data1_15360[(alu1+260)];
  var val68 = data1_15360[(alu1+261)];
  var val69 = data1_15360[(alu1+262)];
  var val70 = data1_15360[(alu1+263)];
  var val71 = data1_15360[(alu1+264)];
  var val72 = data1_15360[(alu1+265)];
  var val73 = data1_15360[(alu1+266)];
  var val74 = data1_15360[(alu1+267)];
  var val75 = data1_15360[(alu1+268)];
  var val76 = data1_15360[(alu1+269)];
  var val77 = data1_15360[(alu1+270)];
  var val78 = data1_15360[(alu1+271)];
  var val79 = data1_15360[(alu1+272)];
  var val80 = data1_15360[(alu1+273)];
  var val81 = data1_15360[(alu1+274)];
  var val82 = data1_15360[(alu1+275)];
  var val83 = data1_15360[(alu1+276)];
  var val84 = data1_15360[(alu1+277)];
  var val85 = data1_15360[(alu1+278)];
  var val86 = data1_15360[(alu1+279)];
  var val87 = data1_15360[(alu1+280)];
  var val88 = data1_15360[(alu1+281)];
  var val89 = data1_15360[(alu1+282)];
  var val90 = data1_15360[(alu1+283)];
  var val91 = data1_15360[(alu1+284)];
  var val92 = data1_15360[(alu1+285)];
  var val93 = data1_15360[(alu1+286)];
  var val94 = data1_15360[(alu1+287)];
  var val95 = data1_15360[alu1];
  var gidx0 = i32(gindex.x); /* 5 */
  var lidx2 = i32(lindex.z); /* 8 */
  var alu2 = ((gidx0*3072)+(lidx2*384)+alu0);
  var val96 = data2_15360[(alu2+1)];
  var val97 = data2_15360[(alu2+2)];
  var val98 = data2_15360[(alu2+3)];
  var val99 = data2_15360[(alu2+4)];
  var val100 = data2_15360[(alu2+5)];
  var val101 = data2_15360[(alu2+6)];
  var val102 = data2_15360[(alu2+7)];
  var val103 = data2_15360[(alu2+8)];
  var val104 = data2_15360[(alu2+9)];
  var val105 = data2_15360[(alu2+10)];
  var val106 = data2_15360[(alu2+11)];
  var val107 = data2_15360[(alu2+12)];
  var val108 = data2_15360[(alu2+13)];
  var val109 = data2_15360[(alu2+14)];
  var val110 = data2_15360[(alu2+15)];
  var val111 = data2_15360[(alu2+16)];
  var val112 = data2_15360[(alu2+17)];
  var val113 = data2_15360[(alu2+18)];
  var val114 = data2_15360[(alu2+19)];
  var val115 = data2_15360[(alu2+20)];
  var val116 = data2_15360[(alu2+21)];
  var val117 = data2_15360[(alu2+22)];
  var val118 = data2_15360[(alu2+23)];
  var val119 = data2_15360[(alu2+24)];
  var val120 = data2_15360[(alu2+25)];
  var val121 = data2_15360[(alu2+26)];
  var val122 = data2_15360[(alu2+27)];
  var val123 = data2_15360[(alu2+28)];
  var val124 = data2_15360[(alu2+29)];
  var val125 = data2_15360[(alu2+30)];
  var val126 = data2_15360[(alu2+31)];
  var val127 = data2_15360[(alu2+128)];
  var val128 = data2_15360[(alu2+129)];
  var val129 = data2_15360[(alu2+130)];
  var val130 = data2_15360[(alu2+131)];
  var val131 = data2_15360[(alu2+132)];
  var val132 = data2_15360[(alu2+133)];
  var val133 = data2_15360[(alu2+134)];
  var val134 = data2_15360[(alu2+135)];
  var val135 = data2_15360[(alu2+136)];
  var val136 = data2_15360[(alu2+137)];
  var val137 = data2_15360[(alu2+138)];
  var val138 = data2_15360[(alu2+139)];
  var val139 = data2_15360[(alu2+140)];
  var val140 = data2_15360[(alu2+141)];
  var val141 = data2_15360[(alu2+142)];
  var val142 = data2_15360[(alu2+143)];
  var val143 = data2_15360[(alu2+144)];
  var val144 = data2_15360[(alu2+145)];
  var val145 = data2_15360[(alu2+146)];
  var val146 = data2_15360[(alu2+147)];
  var val147 = data2_15360[(alu2+148)];
  var val148 = data2_15360[(alu2+149)];
  var val149 = data2_15360[(alu2+150)];
  var val150 = data2_15360[(alu2+151)];
  var val151 = data2_15360[(alu2+152)];
  var val152 = data2_15360[(alu2+153)];
  var val153 = data2_15360[(alu2+154)];
  var val154 = data2_15360[(alu2+155)];
  var val155 = data2_15360[(alu2+156)];
  var val156 = data2_15360[(alu2+157)];
  var val157 = data2_15360[(alu2+158)];
  var val158 = data2_15360[(alu2+159)];
  var val159 = data2_15360[(alu2+256)];
  var val160 = data2_15360[(alu2+257)];
  var val161 = data2_15360[(alu2+258)];
  var val162 = data2_15360[(alu2+259)];
  var val163 = data2_15360[(alu2+260)];
  var val164 = data2_15360[(alu2+261)];
  var val165 = data2_15360[(alu2+262)];
  var val166 = data2_15360[(alu2+263)];
  var val167 = data2_15360[(alu2+264)];
  var val168 = data2_15360[(alu2+265)];
  var val169 = data2_15360[(alu2+266)];
  var val170 = data2_15360[(alu2+267)];
  var val171 = data2_15360[(alu2+268)];
  var val172 = data2_15360[(alu2+269)];
  var val173 = data2_15360[(alu2+270)];
  var val174 = data2_15360[(alu2+271)];
  var val175 = data2_15360[(alu2+272)];
  var val176 = data2_15360[(alu2+273)];
  var val177 = data2_15360[(alu2+274)];
  var val178 = data2_15360[(alu2+275)];
  var val179 = data2_15360[(alu2+276)];
  var val180 = data2_15360[(alu2+277)];
  var val181 = data2_15360[(alu2+278)];
  var val182 = data2_15360[(alu2+279)];
  var val183 = data2_15360[(alu2+280)];
  var val184 = data2_15360[(alu2+281)];
  var val185 = data2_15360[(alu2+282)];
  var val186 = data2_15360[(alu2+283)];
  var val187 = data2_15360[(alu2+284)];
  var val188 = data2_15360[(alu2+285)];
  var val189 = data2_15360[(alu2+286)];
  var val190 = data2_15360[(alu2+287)];
  var val191 = data2_15360[alu2];
  var alu3 = ((gidx0*24)+(lidx2*3)+(gidx1*2880)+(lidx1*360)+(gidx2*28800)+(lidx0*14400));
  data0_57600[(alu3+120)] = (((val31*val191)+(val32*val96)+(val33*val97)+(val34*val98)+(val35*val99)+(val36*val100)+(val37*val101)+(val38*val102)+(val39*val103)+(val40*val104)+(val41*val105)+(val42*val106)+(val43*val107)+(val44*val108)+(val45*val109)+(val46*val110)+(val47*val111)+(val48*val112)+(val49*val113)+(val50*val114)+(val51*val115)+(val52*val116)+(val53*val117)+(val54*val118)+(val55*val119)+(val56*val120)+(val57*val121)+(val58*val122)+(val59*val123)+(val60*val124)+(val61*val125)+(val62*val126))*0.17677669529663687f);
  data0_57600[(alu3+121)] = (((val31*val127)+(val32*val128)+(val33*val129)+(val34*val130)+(val35*val131)+(val36*val132)+(val37*val133)+(val38*val134)+(val39*val135)+(val40*val136)+(val41*val137)+(val42*val138)+(val43*val139)+(val44*val140)+(val45*val141)+(val46*val142)+(val47*val143)+(val48*val144)+(val49*val145)+(val50*val146)+(val51*val147)+(val52*val148)+(val53*val149)+(val54*val150)+(val55*val151)+(val56*val152)+(val57*val153)+(val58*val154)+(val59*val155)+(val60*val156)+(val61*val157)+(val62*val158))*0.17677669529663687f);
  data0_57600[(alu3+122)] = (((val31*val159)+(val32*val160)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val40*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val51*val179)+(val52*val180)+(val53*val181)+(val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val62*val190))*0.17677669529663687f);
  data0_57600[(alu3+240)] = (((val63*val191)+(val64*val96)+(val65*val97)+(val66*val98)+(val67*val99)+(val68*val100)+(val69*val101)+(val70*val102)+(val71*val103)+(val72*val104)+(val73*val105)+(val74*val106)+(val75*val107)+(val76*val108)+(val77*val109)+(val78*val110)+(val79*val111)+(val80*val112)+(val81*val113)+(val82*val114)+(val83*val115)+(val84*val116)+(val85*val117)+(val86*val118)+(val87*val119)+(val88*val120)+(val89*val121)+(val90*val122)+(val91*val123)+(val92*val124)+(val93*val125)+(val94*val126))*0.17677669529663687f);
  data0_57600[(alu3+241)] = (((val63*val127)+(val64*val128)+(val65*val129)+(val66*val130)+(val67*val131)+(val68*val132)+(val69*val133)+(val70*val134)+(val71*val135)+(val72*val136)+(val73*val137)+(val74*val138)+(val75*val139)+(val76*val140)+(val77*val141)+(val78*val142)+(val79*val143)+(val80*val144)+(val81*val145)+(val82*val146)+(val83*val147)+(val84*val148)+(val85*val149)+(val86*val150)+(val87*val151)+(val88*val152)+(val89*val153)+(val90*val154)+(val91*val155)+(val92*val156)+(val93*val157)+(val94*val158))*0.17677669529663687f);
  data0_57600[(alu3+242)] = (((val63*val159)+(val64*val160)+(val65*val161)+(val66*val162)+(val67*val163)+(val68*val164)+(val69*val165)+(val70*val166)+(val71*val167)+(val72*val168)+(val73*val169)+(val74*val170)+(val75*val171)+(val76*val172)+(val77*val173)+(val78*val174)+(val79*val175)+(val80*val176)+(val81*val177)+(val82*val178)+(val83*val179)+(val84*val180)+(val85*val181)+(val86*val182)+(val87*val183)+(val88*val184)+(val89*val185)+(val90*val186)+(val91*val187)+(val92*val188)+(val93*val189)+(val94*val190))*0.17677669529663687f);
  data0_57600[(alu3+1)] = (((val95*val127)+(val0*val128)+(val1*val129)+(val2*val130)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val13*val141)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val24*val152)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158))*0.17677669529663687f);
  data0_57600[(alu3+2)] = (((val95*val159)+(val0*val160)+(val1*val161)+(val2*val162)+(val3*val163)+(val4*val164)+(val5*val165)+(val6*val166)+(val7*val167)+(val8*val168)+(val9*val169)+(val10*val170)+(val11*val171)+(val12*val172)+(val13*val173)+(val14*val174)+(val15*val175)+(val16*val176)+(val17*val177)+(val18*val178)+(val19*val179)+(val20*val180)+(val21*val181)+(val22*val182)+(val23*val183)+(val24*val184)+(val25*val185)+(val26*val186)+(val27*val187)+(val28*val188)+(val29*val189)+(val30*val190))*0.17677669529663687f);
  data0_57600[alu3] = (((val95*val191)+(val0*val96)+(val1*val97)+(val2*val98)+(val3*val99)+(val4*val100)+(val5*val101)+(val6*val102)+(val7*val103)+(val8*val104)+(val9*val105)+(val10*val106)+(val11*val107)+(val12*val108)+(val13*val109)+(val14*val110)+(val15*val111)+(val16*val112)+(val17*val113)+(val18*val114)+(val19*val115)+(val20*val116)+(val21*val117)+(val22*val118)+(val23*val119)+(val24*val120)+(val25*val121)+(val26*val122)+(val27*val123)+(val28*val124)+(val29*val125)+(val30*val126))*0.17677669529663687f);
}`;

const r_15_32_30_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_57600:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 30; Ridx0++) {
    var alu1 = ((gidx0*3840)+(lidx0*120)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_57600[(alu1+1)];
    var val1 = data1_57600[(alu1+2)];
    var val2 = data1_57600[(alu1+3)];
    var val3 = data1_57600[alu1];
    var alu2 = select(acc0[0],val3,(acc0[0]<val3));
    var alu3 = select(alu2,val0,(alu2<val0));
    var alu4 = select(alu3,val1,(alu3<val1));
    var alu5 = select(alu4,val2,(alu4<val2));
    acc0[0] = alu5;
  }
  data0_480[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)))] = acc0[0];
}`;

const r_15_32_30_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_480:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_57600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_480:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<5u)));
  var val0 = data2_480[alu0];
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 30; Ridx0++) {
    var alu2 = ((gidx0*3840)+(lidx0*120)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val1 = data1_57600[(alu2+1)];
    var val2 = data1_57600[(alu2+2)];
    var val3 = data1_57600[(alu2+3)];
    var val4 = data1_57600[alu2];
    acc0[0] = (acc0[0]+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val1-val0)*1.4426950408889634f))+exp2(((val2-val0)*1.4426950408889634f))+exp2(((val3-val0)*1.4426950408889634f)));
  }
  data0_480[alu0] = acc0[0];
}`;

const r_2_5_2_8_8_4_3_30_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_57600:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_480:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_480:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_15360:array<f32>;
@compute @workgroup_size(2,8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((gidx0*24)+(lidx1*3)+(gidx1*240)+(lidx0*120));
  var alu1 = (alu0+1);
  var val0 = data2_480[alu1];
  var alu2 = (alu0+2);
  var val1 = data2_480[alu2];
  var val2 = data2_480[alu0];
  var lidx2 = i32(lindex.z); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(lidx2)<<2u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 30; Ridx0++) {
    var cast1 = bitcast<u32>(Ridx0);
    var alu15 = ((gidx0*2880)+(lidx1*360)+bitcast<i32>((cast1<<2u))+(gidx1*28800)+(lidx0*14400));
    var val3 = data1_57600[alu15];
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<6u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast0+bitcast<i32>((cast1<<9u)));
    var val4 = data4_15360[alu16];
    var val5 = data1_57600[(alu15+1)];
    var val6 = data4_15360[(alu16+128)];
    var val7 = data1_57600[(alu15+2)];
    var val8 = data4_15360[(alu16+256)];
    var val9 = data1_57600[(alu15+3)];
    var val10 = data4_15360[(alu16+384)];
    var val11 = data1_57600[(alu15+120)];
    var val12 = data1_57600[(alu15+121)];
    var val13 = data1_57600[(alu15+122)];
    var val14 = data1_57600[(alu15+123)];
    var val15 = data1_57600[(alu15+240)];
    var val16 = data1_57600[(alu15+241)];
    var val17 = data1_57600[(alu15+242)];
    var val18 = data1_57600[(alu15+243)];
    var val19 = data4_15360[(alu16+1)];
    var val20 = data4_15360[(alu16+2)];
    var val21 = data4_15360[(alu16+129)];
    var val22 = data4_15360[(alu16+257)];
    var val23 = data4_15360[(alu16+385)];
    var val24 = data4_15360[(alu16+130)];
    var val25 = data4_15360[(alu16+258)];
    var val26 = data4_15360[(alu16+386)];
    var val27 = data4_15360[(alu16+3)];
    var val28 = data4_15360[(alu16+131)];
    var val29 = data4_15360[(alu16+259)];
    var val30 = data4_15360[(alu16+387)];
    var alu17 = exp2(((val5-val2)*1.4426950408889634f));
    var alu18 = exp2(((val7-val2)*1.4426950408889634f));
    var alu19 = exp2(((val9-val2)*1.4426950408889634f));
    var alu20 = exp2(((val11-val0)*1.4426950408889634f));
    var alu21 = exp2(((val12-val0)*1.4426950408889634f));
    var alu22 = exp2(((val13-val0)*1.4426950408889634f));
    var alu23 = exp2(((val14-val0)*1.4426950408889634f));
    var alu24 = exp2(((val15-val1)*1.4426950408889634f));
    var alu25 = exp2(((val16-val1)*1.4426950408889634f));
    var alu26 = exp2(((val17-val1)*1.4426950408889634f));
    var alu27 = exp2(((val18-val1)*1.4426950408889634f));
    var alu28 = exp2(((val3-val2)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu28*val4)+(alu17*val6)+(alu18*val8)+(alu19*val10));
    acc0[1] = (acc0[1]+(alu20*val4)+(alu21*val6)+(alu22*val8)+(alu23*val10));
    acc0[2] = (acc0[2]+(alu24*val4)+(alu25*val6)+(alu26*val8)+(alu27*val10));
    acc0[3] = (acc0[3]+(alu28*val19)+(alu17*val21)+(alu18*val22)+(alu19*val23));
    acc0[4] = (acc0[4]+(alu20*val19)+(alu21*val21)+(alu22*val22)+(alu23*val23));
    acc0[5] = (acc0[5]+(alu24*val19)+(alu25*val21)+(alu26*val22)+(alu27*val23));
    acc0[6] = (acc0[6]+(alu28*val20)+(alu17*val24)+(alu18*val25)+(alu19*val26));
    acc0[7] = (acc0[7]+(alu20*val20)+(alu21*val24)+(alu22*val25)+(alu23*val26));
    acc0[8] = (acc0[8]+(alu24*val20)+(alu25*val24)+(alu26*val25)+(alu27*val26));
    acc0[9] = (acc0[9]+(alu28*val27)+(alu17*val28)+(alu18*val29)+(alu19*val30));
    acc0[10] = (acc0[10]+(alu20*val27)+(alu21*val28)+(alu22*val29)+(alu23*val30));
    acc0[11] = (acc0[11]+(alu24*val27)+(alu25*val28)+(alu26*val29)+(alu27*val30));
  }
  var val31 = data3_480[alu0];
  var val32 = data3_480[alu1];
  var val33 = data3_480[alu2];
  var alu42 = ((gidx0*768)+(lidx1*96)+cast0+(gidx1*7680)+(lidx0*3840));
  var alu43 = (1/val32);
  data0_15360[(alu42+32)] = (acc0[1]*alu43);
  data0_15360[(alu42+33)] = (acc0[4]*alu43);
  data0_15360[(alu42+34)] = (acc0[7]*alu43);
  data0_15360[(alu42+35)] = (acc0[10]*alu43);
  var alu48 = (1/val33);
  data0_15360[(alu42+64)] = (acc0[2]*alu48);
  data0_15360[(alu42+65)] = (acc0[5]*alu48);
  data0_15360[(alu42+66)] = (acc0[8]*alu48);
  data0_15360[(alu42+67)] = (acc0[11]*alu48);
  var alu53 = (1/val31);
  data0_15360[(alu42+1)] = (acc0[3]*alu53);
  data0_15360[(alu42+2)] = (acc0[6]*alu53);
  data0_15360[(alu42+3)] = (acc0[9]*alu53);
  data0_15360[alu42] = (acc0[0]*alu53);
}`;

const r_5_2_8_16_4_3_4_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_15360:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_16384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var Ridx0_0 = 0; Ridx0_0 < 4; Ridx0_0++) {
    var alu13 = ((gidx1*768)+(lidx0*96)+(Ridx0_0*3840));
    var val0 = data2_15360[(alu13+2)];
    var val1 = data2_15360[alu13];
    var alu14 = (alu0+bitcast<i32>((bitcast<u32>(Ridx0_0)<<12u)));
    var val2 = data3_16384[alu14];
    var val3 = data2_15360[(alu13+1)];
    var val4 = data3_16384[(alu14+128)];
    var val5 = data3_16384[(alu14+256)];
    var val6 = data2_15360[(alu13+3)];
    var val7 = data3_16384[(alu14+384)];
    var val8 = data2_15360[(alu13+4)];
    var val9 = data3_16384[(alu14+512)];
    var val10 = data2_15360[(alu13+5)];
    var val11 = data3_16384[(alu14+640)];
    var val12 = data2_15360[(alu13+6)];
    var val13 = data3_16384[(alu14+768)];
    var val14 = data2_15360[(alu13+7)];
    var val15 = data3_16384[(alu14+896)];
    var val16 = data2_15360[(alu13+8)];
    var val17 = data3_16384[(alu14+1024)];
    var val18 = data2_15360[(alu13+9)];
    var val19 = data3_16384[(alu14+1152)];
    var val20 = data2_15360[(alu13+10)];
    var val21 = data3_16384[(alu14+1280)];
    var val22 = data2_15360[(alu13+11)];
    var val23 = data3_16384[(alu14+1408)];
    var val24 = data2_15360[(alu13+12)];
    var val25 = data3_16384[(alu14+1536)];
    var val26 = data2_15360[(alu13+13)];
    var val27 = data3_16384[(alu14+1664)];
    var val28 = data2_15360[(alu13+14)];
    var val29 = data3_16384[(alu14+1792)];
    var val30 = data2_15360[(alu13+15)];
    var val31 = data3_16384[(alu14+1920)];
    var val32 = data2_15360[(alu13+16)];
    var val33 = data3_16384[(alu14+2048)];
    var val34 = data2_15360[(alu13+17)];
    var val35 = data3_16384[(alu14+2176)];
    var val36 = data2_15360[(alu13+18)];
    var val37 = data3_16384[(alu14+2304)];
    var val38 = data2_15360[(alu13+19)];
    var val39 = data3_16384[(alu14+2432)];
    var val40 = data2_15360[(alu13+20)];
    var val41 = data3_16384[(alu14+2560)];
    var val42 = data2_15360[(alu13+21)];
    var val43 = data3_16384[(alu14+2688)];
    var val44 = data2_15360[(alu13+22)];
    var val45 = data3_16384[(alu14+2816)];
    var val46 = data2_15360[(alu13+23)];
    var val47 = data3_16384[(alu14+2944)];
    var val48 = data2_15360[(alu13+24)];
    var val49 = data3_16384[(alu14+3072)];
    var val50 = data2_15360[(alu13+25)];
    var val51 = data3_16384[(alu14+3200)];
    var val52 = data2_15360[(alu13+26)];
    var val53 = data3_16384[(alu14+3328)];
    var val54 = data2_15360[(alu13+27)];
    var val55 = data3_16384[(alu14+3456)];
    var val56 = data2_15360[(alu13+28)];
    var val57 = data3_16384[(alu14+3584)];
    var val58 = data2_15360[(alu13+29)];
    var val59 = data3_16384[(alu14+3712)];
    var val60 = data2_15360[(alu13+30)];
    var val61 = data3_16384[(alu14+3840)];
    var val62 = data2_15360[(alu13+31)];
    var val63 = data3_16384[(alu14+3968)];
    var val64 = data2_15360[(alu13+32)];
    var val65 = data2_15360[(alu13+33)];
    var val66 = data2_15360[(alu13+34)];
    var val67 = data2_15360[(alu13+35)];
    var val68 = data2_15360[(alu13+36)];
    var val69 = data2_15360[(alu13+37)];
    var val70 = data2_15360[(alu13+38)];
    var val71 = data2_15360[(alu13+39)];
    var val72 = data2_15360[(alu13+40)];
    var val73 = data2_15360[(alu13+41)];
    var val74 = data2_15360[(alu13+42)];
    var val75 = data2_15360[(alu13+43)];
    var val76 = data2_15360[(alu13+44)];
    var val77 = data2_15360[(alu13+45)];
    var val78 = data2_15360[(alu13+46)];
    var val79 = data2_15360[(alu13+47)];
    var val80 = data2_15360[(alu13+48)];
    var val81 = data2_15360[(alu13+49)];
    var val82 = data2_15360[(alu13+50)];
    var val83 = data2_15360[(alu13+51)];
    var val84 = data2_15360[(alu13+52)];
    var val85 = data2_15360[(alu13+53)];
    var val86 = data2_15360[(alu13+54)];
    var val87 = data2_15360[(alu13+55)];
    var val88 = data2_15360[(alu13+56)];
    var val89 = data2_15360[(alu13+57)];
    var val90 = data2_15360[(alu13+58)];
    var val91 = data2_15360[(alu13+59)];
    var val92 = data2_15360[(alu13+60)];
    var val93 = data2_15360[(alu13+61)];
    var val94 = data2_15360[(alu13+62)];
    var val95 = data2_15360[(alu13+63)];
    var val96 = data2_15360[(alu13+64)];
    var val97 = data2_15360[(alu13+65)];
    var val98 = data2_15360[(alu13+66)];
    var val99 = data2_15360[(alu13+67)];
    var val100 = data2_15360[(alu13+68)];
    var val101 = data2_15360[(alu13+69)];
    var val102 = data2_15360[(alu13+70)];
    var val103 = data2_15360[(alu13+71)];
    var val104 = data2_15360[(alu13+72)];
    var val105 = data2_15360[(alu13+73)];
    var val106 = data2_15360[(alu13+74)];
    var val107 = data2_15360[(alu13+75)];
    var val108 = data2_15360[(alu13+76)];
    var val109 = data2_15360[(alu13+77)];
    var val110 = data2_15360[(alu13+78)];
    var val111 = data2_15360[(alu13+79)];
    var val112 = data2_15360[(alu13+80)];
    var val113 = data2_15360[(alu13+81)];
    var val114 = data2_15360[(alu13+82)];
    var val115 = data2_15360[(alu13+83)];
    var val116 = data2_15360[(alu13+84)];
    var val117 = data2_15360[(alu13+85)];
    var val118 = data2_15360[(alu13+86)];
    var val119 = data2_15360[(alu13+87)];
    var val120 = data2_15360[(alu13+88)];
    var val121 = data2_15360[(alu13+89)];
    var val122 = data2_15360[(alu13+90)];
    var val123 = data2_15360[(alu13+91)];
    var val124 = data2_15360[(alu13+92)];
    var val125 = data2_15360[(alu13+93)];
    var val126 = data2_15360[(alu13+94)];
    var val127 = data2_15360[(alu13+95)];
    var val128 = data3_16384[(alu14+1)];
    var val129 = data3_16384[(alu14+2)];
    var val130 = data3_16384[(alu14+130)];
    var val131 = data3_16384[(alu14+258)];
    var val132 = data3_16384[(alu14+386)];
    var val133 = data3_16384[(alu14+514)];
    var val134 = data3_16384[(alu14+642)];
    var val135 = data3_16384[(alu14+770)];
    var val136 = data3_16384[(alu14+898)];
    var val137 = data3_16384[(alu14+1026)];
    var val138 = data3_16384[(alu14+1154)];
    var val139 = data3_16384[(alu14+1282)];
    var val140 = data3_16384[(alu14+1410)];
    var val141 = data3_16384[(alu14+1538)];
    var val142 = data3_16384[(alu14+3585)];
    var val143 = data3_16384[(alu14+3713)];
    var val144 = data3_16384[(alu14+3841)];
    var val145 = data3_16384[(alu14+3969)];
    var val146 = data3_16384[(alu14+1666)];
    var val147 = data3_16384[(alu14+1794)];
    var val148 = data3_16384[(alu14+1922)];
    var val149 = data3_16384[(alu14+2050)];
    var val150 = data3_16384[(alu14+2178)];
    var val151 = data3_16384[(alu14+2306)];
    var val152 = data3_16384[(alu14+2434)];
    var val153 = data3_16384[(alu14+2562)];
    var val154 = data3_16384[(alu14+2690)];
    var val155 = data3_16384[(alu14+2818)];
    var val156 = data3_16384[(alu14+2946)];
    var val157 = data3_16384[(alu14+3074)];
    var val158 = data3_16384[(alu14+3202)];
    var val159 = data3_16384[(alu14+3330)];
    var val160 = data3_16384[(alu14+3458)];
    var val161 = data3_16384[(alu14+3586)];
    var val162 = data3_16384[(alu14+3714)];
    var val163 = data3_16384[(alu14+3842)];
    var val164 = data3_16384[(alu14+3970)];
    var val165 = data3_16384[(alu14+3)];
    var val166 = data3_16384[(alu14+129)];
    var val167 = data3_16384[(alu14+131)];
    var val168 = data3_16384[(alu14+257)];
    var val169 = data3_16384[(alu14+259)];
    var val170 = data3_16384[(alu14+385)];
    var val171 = data3_16384[(alu14+387)];
    var val172 = data3_16384[(alu14+513)];
    var val173 = data3_16384[(alu14+515)];
    var val174 = data3_16384[(alu14+641)];
    var val175 = data3_16384[(alu14+643)];
    var val176 = data3_16384[(alu14+769)];
    var val177 = data3_16384[(alu14+771)];
    var val178 = data3_16384[(alu14+897)];
    var val179 = data3_16384[(alu14+899)];
    var val180 = data3_16384[(alu14+1025)];
    var val181 = data3_16384[(alu14+1027)];
    var val182 = data3_16384[(alu14+1153)];
    var val183 = data3_16384[(alu14+1155)];
    var val184 = data3_16384[(alu14+1281)];
    var val185 = data3_16384[(alu14+1283)];
    var val186 = data3_16384[(alu14+1409)];
    var val187 = data3_16384[(alu14+1411)];
    var val188 = data3_16384[(alu14+1537)];
    var val189 = data3_16384[(alu14+1539)];
    var val190 = data3_16384[(alu14+1665)];
    var val191 = data3_16384[(alu14+1667)];
    var val192 = data3_16384[(alu14+1793)];
    var val193 = data3_16384[(alu14+1795)];
    var val194 = data3_16384[(alu14+1921)];
    var val195 = data3_16384[(alu14+1923)];
    var val196 = data3_16384[(alu14+2049)];
    var val197 = data3_16384[(alu14+2051)];
    var val198 = data3_16384[(alu14+2177)];
    var val199 = data3_16384[(alu14+2179)];
    var val200 = data3_16384[(alu14+2305)];
    var val201 = data3_16384[(alu14+2307)];
    var val202 = data3_16384[(alu14+2433)];
    var val203 = data3_16384[(alu14+2435)];
    var val204 = data3_16384[(alu14+2561)];
    var val205 = data3_16384[(alu14+2563)];
    var val206 = data3_16384[(alu14+2689)];
    var val207 = data3_16384[(alu14+2691)];
    var val208 = data3_16384[(alu14+2817)];
    var val209 = data3_16384[(alu14+2819)];
    var val210 = data3_16384[(alu14+2945)];
    var val211 = data3_16384[(alu14+2947)];
    var val212 = data3_16384[(alu14+3073)];
    var val213 = data3_16384[(alu14+3075)];
    var val214 = data3_16384[(alu14+3201)];
    var val215 = data3_16384[(alu14+3203)];
    var val216 = data3_16384[(alu14+3329)];
    var val217 = data3_16384[(alu14+3331)];
    var val218 = data3_16384[(alu14+3457)];
    var val219 = data3_16384[(alu14+3459)];
    var val220 = data3_16384[(alu14+3587)];
    var val221 = data3_16384[(alu14+3715)];
    var val222 = data3_16384[(alu14+3843)];
    var val223 = data3_16384[(alu14+3971)];
    acc0[0] = (acc0[0]+(val1*val2)+(val3*val4)+(val0*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17)+(val18*val19)+(val20*val21)+(val22*val23)+(val24*val25)+(val26*val27)+(val28*val29)+(val30*val31)+(val32*val33)+(val34*val35)+(val36*val37)+(val38*val39)+(val40*val41)+(val42*val43)+(val44*val45)+(val46*val47)+(val48*val49)+(val50*val51)+(val52*val53)+(val54*val55)+(val56*val57)+(val58*val59)+(val60*val61)+(val62*val63));
    acc0[1] = (acc0[1]+(val64*val2)+(val65*val4)+(val66*val5)+(val67*val7)+(val68*val9)+(val69*val11)+(val70*val13)+(val71*val15)+(val72*val17)+(val73*val19)+(val74*val21)+(val75*val23)+(val76*val25)+(val77*val27)+(val78*val29)+(val79*val31)+(val80*val33)+(val81*val35)+(val82*val37)+(val83*val39)+(val84*val41)+(val85*val43)+(val86*val45)+(val87*val47)+(val88*val49)+(val89*val51)+(val90*val53)+(val91*val55)+(val92*val57)+(val93*val59)+(val94*val61)+(val95*val63));
    acc0[2] = (acc0[2]+(val96*val2)+(val97*val4)+(val98*val5)+(val99*val7)+(val100*val9)+(val101*val11)+(val102*val13)+(val103*val15)+(val104*val17)+(val105*val19)+(val106*val21)+(val107*val23)+(val108*val25)+(val109*val27)+(val110*val29)+(val111*val31)+(val112*val33)+(val113*val35)+(val114*val37)+(val115*val39)+(val116*val41)+(val117*val43)+(val118*val45)+(val119*val47)+(val120*val49)+(val121*val51)+(val122*val53)+(val123*val55)+(val124*val57)+(val125*val59)+(val126*val61)+(val127*val63));
    acc0[3] = (acc0[3]+(val1*val128)+(val3*val166)+(val0*val168)+(val6*val170)+(val8*val172)+(val10*val174)+(val12*val176)+(val14*val178)+(val16*val180)+(val18*val182)+(val20*val184)+(val22*val186)+(val24*val188)+(val26*val190)+(val28*val192)+(val30*val194)+(val32*val196)+(val34*val198)+(val36*val200)+(val38*val202)+(val40*val204)+(val42*val206)+(val44*val208)+(val46*val210)+(val48*val212)+(val50*val214)+(val52*val216)+(val54*val218)+(val56*val142)+(val58*val143)+(val60*val144)+(val62*val145));
    acc0[4] = (acc0[4]+(val64*val128)+(val65*val166)+(val66*val168)+(val67*val170)+(val68*val172)+(val69*val174)+(val70*val176)+(val71*val178)+(val72*val180)+(val73*val182)+(val74*val184)+(val75*val186)+(val76*val188)+(val77*val190)+(val78*val192)+(val79*val194)+(val80*val196)+(val81*val198)+(val82*val200)+(val83*val202)+(val84*val204)+(val85*val206)+(val86*val208)+(val87*val210)+(val88*val212)+(val89*val214)+(val90*val216)+(val91*val218)+(val92*val142)+(val93*val143)+(val94*val144)+(val95*val145));
    acc0[5] = (acc0[5]+(val96*val128)+(val97*val166)+(val98*val168)+(val99*val170)+(val100*val172)+(val101*val174)+(val102*val176)+(val103*val178)+(val104*val180)+(val105*val182)+(val106*val184)+(val107*val186)+(val108*val188)+(val109*val190)+(val110*val192)+(val111*val194)+(val112*val196)+(val113*val198)+(val114*val200)+(val115*val202)+(val116*val204)+(val117*val206)+(val118*val208)+(val119*val210)+(val120*val212)+(val121*val214)+(val122*val216)+(val123*val218)+(val124*val142)+(val125*val143)+(val126*val144)+(val127*val145));
    acc0[6] = (acc0[6]+(val1*val129)+(val3*val130)+(val0*val131)+(val6*val132)+(val8*val133)+(val10*val134)+(val12*val135)+(val14*val136)+(val16*val137)+(val18*val138)+(val20*val139)+(val22*val140)+(val24*val141)+(val26*val146)+(val28*val147)+(val30*val148)+(val32*val149)+(val34*val150)+(val36*val151)+(val38*val152)+(val40*val153)+(val42*val154)+(val44*val155)+(val46*val156)+(val48*val157)+(val50*val158)+(val52*val159)+(val54*val160)+(val56*val161)+(val58*val162)+(val60*val163)+(val62*val164));
    acc0[7] = (acc0[7]+(val64*val129)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val73*val138)+(val74*val139)+(val75*val140)+(val76*val141)+(val77*val146)+(val78*val147)+(val79*val148)+(val80*val149)+(val81*val150)+(val82*val151)+(val83*val152)+(val84*val153)+(val85*val154)+(val86*val155)+(val87*val156)+(val88*val157)+(val89*val158)+(val90*val159)+(val91*val160)+(val92*val161)+(val93*val162)+(val94*val163)+(val95*val164));
    acc0[8] = (acc0[8]+(val96*val129)+(val97*val130)+(val98*val131)+(val99*val132)+(val100*val133)+(val101*val134)+(val102*val135)+(val103*val136)+(val104*val137)+(val105*val138)+(val106*val139)+(val107*val140)+(val108*val141)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val152)+(val116*val153)+(val117*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160)+(val124*val161)+(val125*val162)+(val126*val163)+(val127*val164));
    acc0[9] = (acc0[9]+(val1*val165)+(val3*val167)+(val0*val169)+(val6*val171)+(val8*val173)+(val10*val175)+(val12*val177)+(val14*val179)+(val16*val181)+(val18*val183)+(val20*val185)+(val22*val187)+(val24*val189)+(val26*val191)+(val28*val193)+(val30*val195)+(val32*val197)+(val34*val199)+(val36*val201)+(val38*val203)+(val40*val205)+(val42*val207)+(val44*val209)+(val46*val211)+(val48*val213)+(val50*val215)+(val52*val217)+(val54*val219)+(val56*val220)+(val58*val221)+(val60*val222)+(val62*val223));
    acc0[10] = (acc0[10]+(val64*val165)+(val65*val167)+(val66*val169)+(val67*val171)+(val68*val173)+(val69*val175)+(val70*val177)+(val71*val179)+(val72*val181)+(val73*val183)+(val74*val185)+(val75*val187)+(val76*val189)+(val77*val191)+(val78*val193)+(val79*val195)+(val80*val197)+(val81*val199)+(val82*val201)+(val83*val203)+(val84*val205)+(val85*val207)+(val86*val209)+(val87*val211)+(val88*val213)+(val89*val215)+(val90*val217)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val95*val223));
    acc0[11] = (acc0[11]+(val96*val165)+(val97*val167)+(val98*val169)+(val99*val171)+(val100*val173)+(val101*val175)+(val102*val177)+(val103*val179)+(val104*val181)+(val105*val183)+(val106*val185)+(val107*val187)+(val108*val189)+(val109*val191)+(val110*val193)+(val111*val195)+(val112*val197)+(val113*val199)+(val114*val201)+(val115*val203)+(val116*val205)+(val117*val207)+(val118*val209)+(val119*val211)+(val120*val213)+(val121*val215)+(val122*val217)+(val123*val219)+(val124*val220)+(val125*val221)+(val126*val222)+(val127*val223));
  }
  var alu28 = (alu0+(gidx1*3072)+(lidx0*384));
  var val224 = data1_15360[alu28];
  var val225 = data4_128[alu0];
  var alu29 = (alu28+1);
  var val226 = data1_15360[alu29];
  var val227 = data4_128[(alu0+1)];
  var alu30 = (alu28+2);
  var val228 = data1_15360[alu30];
  var val229 = data4_128[(alu0+2)];
  var alu31 = (alu28+3);
  var val230 = data1_15360[alu31];
  var val231 = data4_128[(alu0+3)];
  var alu32 = (alu28+128);
  var val232 = data1_15360[alu32];
  var alu33 = (alu28+129);
  var val233 = data1_15360[alu33];
  var alu34 = (alu28+130);
  var val234 = data1_15360[alu34];
  var alu35 = (alu28+131);
  var val235 = data1_15360[alu35];
  var alu36 = (alu28+256);
  var val236 = data1_15360[alu36];
  var alu37 = (alu28+257);
  var val237 = data1_15360[alu37];
  var alu38 = (alu28+258);
  var val238 = data1_15360[alu38];
  var alu39 = (alu28+259);
  var val239 = data1_15360[alu39];
  data0_15360[alu28] = (val224+acc0[0]+val225);
  data0_15360[alu29] = (val226+acc0[3]+val227);
  data0_15360[alu30] = (val228+acc0[6]+val229);
  data0_15360[alu31] = (val230+acc0[9]+val231);
  data0_15360[alu32] = (val232+acc0[1]+val225);
  data0_15360[alu33] = (val233+acc0[4]+val227);
  data0_15360[alu34] = (val234+acc0[7]+val229);
  data0_15360[alu35] = (val235+acc0[10]+val231);
  data0_15360[alu36] = (val236+acc0[2]+val225);
  data0_15360[alu37] = (val237+acc0[5]+val227);
  data0_15360[alu38] = (val238+acc0[8]+val229);
  data0_15360[alu39] = (val239+acc0[11]+val231);
}`;

const r_120_16_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_120:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 120 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val0 = data1_15360[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
    acc0[0] = (acc0[0]+val0);
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val1 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val1);
  }
  var alu8 = ((bool(lidx0))!=true);
  if (alu8) {
    data0_120[gidx0] = (acc1[0]*0.0078125f);
  }
}`;

const r_120_16_8n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_120:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_120:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 120 */
  var val0 = data2_120[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val1 = data1_15360[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
    var alu1 = (val1-val0);
    acc0[0] = (acc0[0]+(alu1*alu1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val2 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val2);
  }
  var alu9 = ((bool(lidx0))!=true);
  if (alu9) {
    data0_120[gidx0] = (1/sqrt(((acc1[0]*0.0078125f)+1e-05f)));
  }
}`;

const E_5_2_8_16_4_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_120:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_120:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+(gidx1*3072)+(lidx0*384));
  var val0 = data1_15360[alu1];
  var alu2 = ((gidx1*24)+(lidx0*3));
  var val1 = data2_120[alu2];
  var val2 = data3_120[alu2];
  var val3 = data4_128[alu0];
  var alu3 = (alu0+1);
  var val4 = data4_128[alu3];
  var val5 = data5_128[alu0];
  var alu4 = (alu1+1);
  var val6 = data1_15360[alu4];
  var val7 = data5_128[alu3];
  var alu5 = (alu1+2);
  var val8 = data1_15360[alu5];
  var alu6 = (alu0+2);
  var val9 = data4_128[alu6];
  var val10 = data5_128[alu6];
  var alu7 = (alu1+3);
  var val11 = data1_15360[alu7];
  var alu8 = (alu0+3);
  var val12 = data4_128[alu8];
  var val13 = data5_128[alu8];
  var alu9 = (alu1+128);
  var val14 = data1_15360[alu9];
  var alu10 = (alu2+1);
  var val15 = data2_120[alu10];
  var val16 = data3_120[alu10];
  var alu11 = (alu1+129);
  var val17 = data1_15360[alu11];
  var alu12 = (alu1+130);
  var val18 = data1_15360[alu12];
  var alu13 = (alu1+131);
  var val19 = data1_15360[alu13];
  var alu14 = (alu1+256);
  var val20 = data1_15360[alu14];
  var alu15 = (alu2+2);
  var val21 = data2_120[alu15];
  var val22 = data3_120[alu15];
  var alu16 = (alu1+257);
  var val23 = data1_15360[alu16];
  var alu17 = (alu1+258);
  var val24 = data1_15360[alu17];
  var alu18 = (alu1+259);
  var val25 = data1_15360[alu18];
  data0_15360[alu1] = (((val0-val1)*val2*val3)+val5);
  data0_15360[alu4] = (((val6-val1)*val2*val4)+val7);
  data0_15360[alu5] = (((val8-val1)*val2*val9)+val10);
  data0_15360[alu7] = (((val11-val1)*val2*val12)+val13);
  data0_15360[alu9] = (((val14-val15)*val16*val3)+val5);
  data0_15360[alu11] = (((val17-val15)*val16*val4)+val7);
  data0_15360[alu12] = (((val18-val15)*val16*val9)+val10);
  data0_15360[alu13] = (((val19-val15)*val16*val12)+val13);
  data0_15360[alu14] = (((val20-val21)*val22*val3)+val5);
  data0_15360[alu16] = (((val23-val21)*val22*val4)+val7);
  data0_15360[alu17] = (((val24-val21)*val22*val9)+val10);
  data0_15360[alu18] = (((val25-val21)*val22*val12)+val13);
}`;

const r_15_8_8_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3840:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_4096:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  var cast2 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var cast3 = bitcast<u32>(Ridx0);
    var alu4 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast1<<7u))+bitcast<i32>((cast3<<2u)));
    var val0 = data1_15360[alu4];
    var alu5 = (cast2+bitcast<i32>((cast3<<7u)));
    var val1 = data2_4096[alu5];
    var val2 = data1_15360[(alu4+1)];
    var val3 = data2_4096[(alu5+3)];
    var val4 = data2_4096[(alu5+32)];
    var val5 = data1_15360[(alu4+2)];
    var val6 = data2_4096[(alu5+1)];
    var val7 = data2_4096[(alu5+64)];
    var val8 = data1_15360[(alu4+3)];
    var val9 = data2_4096[(alu5+33)];
    var val10 = data2_4096[(alu5+96)];
    var val11 = data2_4096[(alu5+65)];
    var val12 = data2_4096[(alu5+97)];
    var val13 = data2_4096[(alu5+2)];
    var val14 = data2_4096[(alu5+34)];
    var val15 = data2_4096[(alu5+35)];
    var val16 = data2_4096[(alu5+66)];
    var val17 = data2_4096[(alu5+98)];
    var val18 = data2_4096[(alu5+67)];
    var val19 = data2_4096[(alu5+99)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val4)+(val5*val7)+(val8*val10));
    acc0[1] = (acc0[1]+(val0*val6)+(val2*val9)+(val5*val11)+(val8*val12));
    acc0[2] = (acc0[2]+(val0*val13)+(val2*val14)+(val5*val16)+(val8*val17));
    acc0[3] = (acc0[3]+(val0*val3)+(val2*val15)+(val5*val18)+(val8*val19));
  }
  var val20 = data3_32[cast2];
  var val21 = data3_32[(cast2+1)];
  var val22 = data3_32[(cast2+2)];
  var val23 = data3_32[(cast2+3)];
  var alu11 = (bitcast<i32>((cast0<<8u))+bitcast<i32>((cast1<<5u))+cast2);
  var alu12 = (acc0[0]+val20);
  var alu13 = (acc0[1]+val21);
  var alu14 = (acc0[2]+val22);
  var alu15 = (acc0[3]+val23);
  var alu16 = select(0.0f,alu12,(0.0f<alu12));
  var alu17 = select(0.0f,alu13,(0.0f<alu13));
  var alu18 = select(0.0f,alu14,(0.0f<alu14));
  var alu19 = select(0.0f,alu15,(0.0f<alu15));
  data0_3840[alu11] = alu16;
  data0_3840[(alu11+1)] = alu17;
  data0_3840[(alu11+2)] = alu18;
  data0_3840[(alu11+3)] = alu19;
}`;

const r_5_2_8_16_4_3_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_15360:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_3840:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_4096:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+(gidx1*3072)+(lidx0*384));
  var val0 = data1_15360[alu1];
  var alu2 = ((gidx1*768)+(lidx0*96));
  var val1 = data2_3840[(alu2+1)];
  var val2 = data2_3840[(alu2+20)];
  var val3 = data2_3840[alu2];
  var val4 = data3_4096[alu0];
  var alu3 = (alu0+2);
  var val5 = data3_4096[alu3];
  var alu4 = (alu0+3);
  var val6 = data3_4096[alu4];
  var val7 = data3_4096[(alu0+128)];
  var val8 = data2_3840[(alu2+2)];
  var val9 = data3_4096[(alu0+130)];
  var val10 = data3_4096[(alu0+256)];
  var val11 = data2_3840[(alu2+3)];
  var val12 = data3_4096[(alu0+258)];
  var val13 = data3_4096[(alu0+384)];
  var val14 = data2_3840[(alu2+4)];
  var val15 = data3_4096[(alu0+386)];
  var val16 = data3_4096[(alu0+512)];
  var val17 = data2_3840[(alu2+5)];
  var val18 = data3_4096[(alu0+514)];
  var val19 = data3_4096[(alu0+640)];
  var val20 = data2_3840[(alu2+6)];
  var val21 = data3_4096[(alu0+642)];
  var val22 = data3_4096[(alu0+768)];
  var val23 = data2_3840[(alu2+7)];
  var val24 = data3_4096[(alu0+770)];
  var val25 = data3_4096[(alu0+771)];
  var val26 = data3_4096[(alu0+896)];
  var val27 = data2_3840[(alu2+8)];
  var val28 = data3_4096[(alu0+898)];
  var val29 = data3_4096[(alu0+899)];
  var val30 = data3_4096[(alu0+1024)];
  var val31 = data2_3840[(alu2+9)];
  var val32 = data3_4096[(alu0+1026)];
  var val33 = data3_4096[(alu0+1027)];
  var val34 = data3_4096[(alu0+1152)];
  var val35 = data2_3840[(alu2+10)];
  var val36 = data3_4096[(alu0+1154)];
  var val37 = data3_4096[(alu0+1155)];
  var val38 = data3_4096[(alu0+1280)];
  var val39 = data2_3840[(alu2+11)];
  var val40 = data3_4096[(alu0+1282)];
  var val41 = data3_4096[(alu0+1283)];
  var val42 = data3_4096[(alu0+1408)];
  var val43 = data2_3840[(alu2+12)];
  var val44 = data3_4096[(alu0+1410)];
  var val45 = data3_4096[(alu0+1411)];
  var val46 = data3_4096[(alu0+1536)];
  var val47 = data2_3840[(alu2+13)];
  var val48 = data3_4096[(alu0+1538)];
  var val49 = data3_4096[(alu0+1539)];
  var val50 = data3_4096[(alu0+1664)];
  var val51 = data2_3840[(alu2+14)];
  var val52 = data3_4096[(alu0+1666)];
  var val53 = data3_4096[(alu0+1667)];
  var val54 = data3_4096[(alu0+1792)];
  var val55 = data2_3840[(alu2+15)];
  var val56 = data3_4096[(alu0+1794)];
  var val57 = data3_4096[(alu0+1795)];
  var val58 = data3_4096[(alu0+1920)];
  var val59 = data2_3840[(alu2+16)];
  var val60 = data3_4096[(alu0+1922)];
  var val61 = data3_4096[(alu0+1923)];
  var val62 = data3_4096[(alu0+2048)];
  var val63 = data2_3840[(alu2+17)];
  var val64 = data3_4096[(alu0+2049)];
  var val65 = data3_4096[(alu0+2050)];
  var val66 = data3_4096[(alu0+2051)];
  var val67 = data3_4096[(alu0+2176)];
  var val68 = data2_3840[(alu2+18)];
  var val69 = data3_4096[(alu0+2177)];
  var val70 = data3_4096[(alu0+2178)];
  var val71 = data3_4096[(alu0+2179)];
  var val72 = data3_4096[(alu0+2304)];
  var val73 = data2_3840[(alu2+19)];
  var val74 = data3_4096[(alu0+2305)];
  var val75 = data3_4096[(alu0+2306)];
  var val76 = data3_4096[(alu0+2307)];
  var val77 = data3_4096[(alu0+2432)];
  var val78 = data3_4096[(alu0+2433)];
  var val79 = data3_4096[(alu0+2434)];
  var val80 = data3_4096[(alu0+2435)];
  var val81 = data3_4096[(alu0+2560)];
  var val82 = data2_3840[(alu2+21)];
  var val83 = data3_4096[(alu0+2561)];
  var val84 = data3_4096[(alu0+2562)];
  var val85 = data3_4096[(alu0+2563)];
  var val86 = data3_4096[(alu0+2688)];
  var val87 = data2_3840[(alu2+22)];
  var val88 = data3_4096[(alu0+2689)];
  var val89 = data3_4096[(alu0+2690)];
  var val90 = data3_4096[(alu0+2691)];
  var val91 = data3_4096[(alu0+2816)];
  var val92 = data2_3840[(alu2+23)];
  var val93 = data3_4096[(alu0+2817)];
  var val94 = data3_4096[(alu0+2818)];
  var val95 = data3_4096[(alu0+2819)];
  var val96 = data3_4096[(alu0+2944)];
  var val97 = data2_3840[(alu2+24)];
  var val98 = data3_4096[(alu0+2945)];
  var val99 = data3_4096[(alu0+2946)];
  var val100 = data3_4096[(alu0+2947)];
  var val101 = data3_4096[(alu0+3072)];
  var val102 = data2_3840[(alu2+25)];
  var val103 = data3_4096[(alu0+3073)];
  var val104 = data3_4096[(alu0+3074)];
  var val105 = data3_4096[(alu0+3200)];
  var val106 = data2_3840[(alu2+26)];
  var val107 = data3_4096[(alu0+3201)];
  var val108 = data3_4096[(alu0+3202)];
  var val109 = data3_4096[(alu0+3328)];
  var val110 = data2_3840[(alu2+27)];
  var val111 = data3_4096[(alu0+3329)];
  var val112 = data3_4096[(alu0+3330)];
  var val113 = data3_4096[(alu0+3456)];
  var val114 = data2_3840[(alu2+28)];
  var val115 = data3_4096[(alu0+3457)];
  var val116 = data3_4096[(alu0+3458)];
  var val117 = data3_4096[(alu0+3584)];
  var val118 = data2_3840[(alu2+29)];
  var val119 = data3_4096[(alu0+3585)];
  var val120 = data3_4096[(alu0+3586)];
  var val121 = data3_4096[(alu0+3712)];
  var val122 = data2_3840[(alu2+30)];
  var val123 = data2_3840[(alu2+31)];
  var val124 = data3_4096[(alu0+897)];
  var val125 = data3_4096[(alu0+1025)];
  var val126 = data3_4096[(alu0+1153)];
  var val127 = data3_4096[(alu0+1281)];
  var val128 = data3_4096[(alu0+1409)];
  var val129 = data3_4096[(alu0+1537)];
  var val130 = data3_4096[(alu0+1665)];
  var val131 = data3_4096[(alu0+1793)];
  var val132 = data3_4096[(alu0+1921)];
  var val133 = data3_4096[(alu0+3968)];
  var val134 = data4_128[alu0];
  var alu5 = (alu1+1);
  var val135 = data1_15360[alu5];
  var alu6 = (alu0+1);
  var val136 = data3_4096[alu6];
  var val137 = data3_4096[(alu0+129)];
  var val138 = data3_4096[(alu0+257)];
  var val139 = data3_4096[(alu0+385)];
  var val140 = data3_4096[(alu0+513)];
  var val141 = data3_4096[(alu0+641)];
  var val142 = data3_4096[(alu0+769)];
  var val143 = data3_4096[(alu0+3075)];
  var val144 = data3_4096[(alu0+3203)];
  var val145 = data3_4096[(alu0+3331)];
  var val146 = data3_4096[(alu0+3459)];
  var val147 = data3_4096[(alu0+3587)];
  var val148 = data3_4096[(alu0+3713)];
  var val149 = data3_4096[(alu0+3714)];
  var val150 = data3_4096[(alu0+3715)];
  var val151 = data3_4096[(alu0+3840)];
  var val152 = data3_4096[(alu0+3841)];
  var val153 = data3_4096[(alu0+3842)];
  var val154 = data3_4096[(alu0+3843)];
  var val155 = data3_4096[(alu0+3969)];
  var val156 = data4_128[alu6];
  var alu7 = (alu1+2);
  var val157 = data1_15360[alu7];
  var val158 = data3_4096[(alu0+3970)];
  var val159 = data4_128[alu3];
  var alu8 = (alu1+3);
  var val160 = data1_15360[alu8];
  var val161 = data3_4096[(alu0+131)];
  var val162 = data3_4096[(alu0+259)];
  var val163 = data3_4096[(alu0+387)];
  var val164 = data3_4096[(alu0+515)];
  var val165 = data3_4096[(alu0+643)];
  var val166 = data3_4096[(alu0+3971)];
  var val167 = data4_128[alu4];
  var alu9 = (alu1+128);
  var val168 = data1_15360[alu9];
  var val169 = data2_3840[(alu2+32)];
  var val170 = data2_3840[(alu2+33)];
  var val171 = data2_3840[(alu2+34)];
  var val172 = data2_3840[(alu2+35)];
  var val173 = data2_3840[(alu2+36)];
  var val174 = data2_3840[(alu2+37)];
  var val175 = data2_3840[(alu2+38)];
  var val176 = data2_3840[(alu2+39)];
  var val177 = data2_3840[(alu2+40)];
  var val178 = data2_3840[(alu2+41)];
  var val179 = data2_3840[(alu2+42)];
  var val180 = data2_3840[(alu2+43)];
  var val181 = data2_3840[(alu2+44)];
  var val182 = data2_3840[(alu2+45)];
  var val183 = data2_3840[(alu2+46)];
  var val184 = data2_3840[(alu2+47)];
  var val185 = data2_3840[(alu2+48)];
  var val186 = data2_3840[(alu2+49)];
  var val187 = data2_3840[(alu2+50)];
  var val188 = data2_3840[(alu2+51)];
  var val189 = data2_3840[(alu2+52)];
  var val190 = data2_3840[(alu2+53)];
  var val191 = data2_3840[(alu2+54)];
  var val192 = data2_3840[(alu2+55)];
  var val193 = data2_3840[(alu2+56)];
  var val194 = data2_3840[(alu2+57)];
  var val195 = data2_3840[(alu2+58)];
  var val196 = data2_3840[(alu2+59)];
  var val197 = data2_3840[(alu2+60)];
  var val198 = data2_3840[(alu2+61)];
  var val199 = data2_3840[(alu2+62)];
  var val200 = data2_3840[(alu2+63)];
  var alu10 = (alu1+129);
  var val201 = data1_15360[alu10];
  var alu11 = (alu1+130);
  var val202 = data1_15360[alu11];
  var alu12 = (alu1+131);
  var val203 = data1_15360[alu12];
  var alu13 = (alu1+256);
  var val204 = data1_15360[alu13];
  var val205 = data2_3840[(alu2+64)];
  var val206 = data2_3840[(alu2+65)];
  var val207 = data2_3840[(alu2+66)];
  var val208 = data2_3840[(alu2+67)];
  var val209 = data2_3840[(alu2+68)];
  var val210 = data2_3840[(alu2+69)];
  var val211 = data2_3840[(alu2+70)];
  var val212 = data2_3840[(alu2+71)];
  var val213 = data2_3840[(alu2+72)];
  var val214 = data2_3840[(alu2+73)];
  var val215 = data2_3840[(alu2+74)];
  var val216 = data2_3840[(alu2+75)];
  var val217 = data2_3840[(alu2+76)];
  var val218 = data2_3840[(alu2+77)];
  var val219 = data2_3840[(alu2+78)];
  var val220 = data2_3840[(alu2+79)];
  var val221 = data2_3840[(alu2+80)];
  var val222 = data2_3840[(alu2+81)];
  var val223 = data2_3840[(alu2+82)];
  var val224 = data2_3840[(alu2+83)];
  var val225 = data2_3840[(alu2+84)];
  var val226 = data2_3840[(alu2+85)];
  var val227 = data2_3840[(alu2+86)];
  var val228 = data2_3840[(alu2+87)];
  var val229 = data2_3840[(alu2+88)];
  var val230 = data2_3840[(alu2+89)];
  var val231 = data2_3840[(alu2+90)];
  var val232 = data2_3840[(alu2+91)];
  var val233 = data2_3840[(alu2+92)];
  var val234 = data2_3840[(alu2+93)];
  var val235 = data2_3840[(alu2+94)];
  var val236 = data2_3840[(alu2+95)];
  var alu14 = (alu1+257);
  var val237 = data1_15360[alu14];
  var alu15 = (alu1+258);
  var val238 = data1_15360[alu15];
  var alu16 = (alu1+259);
  var val239 = data1_15360[alu16];
  data0_15360[alu1] = (val0+(val3*val4)+(val1*val7)+(val8*val10)+(val11*val13)+(val14*val16)+(val17*val19)+(val20*val22)+(val23*val26)+(val27*val30)+(val31*val34)+(val35*val38)+(val39*val42)+(val43*val46)+(val47*val50)+(val51*val54)+(val55*val58)+(val59*val62)+(val63*val67)+(val68*val72)+(val73*val77)+(val2*val81)+(val82*val86)+(val87*val91)+(val92*val96)+(val97*val101)+(val102*val105)+(val106*val109)+(val110*val113)+(val114*val117)+(val118*val121)+(val122*val151)+(val123*val133)+val134);
  data0_15360[alu5] = (val135+(val3*val136)+(val1*val137)+(val8*val138)+(val11*val139)+(val14*val140)+(val17*val141)+(val20*val142)+(val23*val124)+(val27*val125)+(val31*val126)+(val35*val127)+(val39*val128)+(val43*val129)+(val47*val130)+(val51*val131)+(val55*val132)+(val59*val64)+(val63*val69)+(val68*val74)+(val73*val78)+(val2*val83)+(val82*val88)+(val87*val93)+(val92*val98)+(val97*val103)+(val102*val107)+(val106*val111)+(val110*val115)+(val114*val119)+(val118*val148)+(val122*val152)+(val123*val155)+val156);
  data0_15360[alu7] = (val157+(val3*val5)+(val1*val9)+(val8*val12)+(val11*val15)+(val14*val18)+(val17*val21)+(val20*val24)+(val23*val28)+(val27*val32)+(val31*val36)+(val35*val40)+(val39*val44)+(val43*val48)+(val47*val52)+(val51*val56)+(val55*val60)+(val59*val65)+(val63*val70)+(val68*val75)+(val73*val79)+(val2*val84)+(val82*val89)+(val87*val94)+(val92*val99)+(val97*val104)+(val102*val108)+(val106*val112)+(val110*val116)+(val114*val120)+(val118*val149)+(val122*val153)+(val123*val158)+val159);
  data0_15360[alu8] = (val160+(val3*val6)+(val1*val161)+(val8*val162)+(val11*val163)+(val14*val164)+(val17*val165)+(val20*val25)+(val23*val29)+(val27*val33)+(val31*val37)+(val35*val41)+(val39*val45)+(val43*val49)+(val47*val53)+(val51*val57)+(val55*val61)+(val59*val66)+(val63*val71)+(val68*val76)+(val73*val80)+(val2*val85)+(val82*val90)+(val87*val95)+(val92*val100)+(val97*val143)+(val102*val144)+(val106*val145)+(val110*val146)+(val114*val147)+(val118*val150)+(val122*val154)+(val123*val166)+val167);
  data0_15360[alu9] = (val168+(val169*val4)+(val170*val7)+(val171*val10)+(val172*val13)+(val173*val16)+(val174*val19)+(val175*val22)+(val176*val26)+(val177*val30)+(val178*val34)+(val179*val38)+(val180*val42)+(val181*val46)+(val182*val50)+(val183*val54)+(val184*val58)+(val185*val62)+(val186*val67)+(val187*val72)+(val188*val77)+(val189*val81)+(val190*val86)+(val191*val91)+(val192*val96)+(val193*val101)+(val194*val105)+(val195*val109)+(val196*val113)+(val197*val117)+(val198*val121)+(val199*val151)+(val200*val133)+val134);
  data0_15360[alu10] = (val201+(val169*val136)+(val170*val137)+(val171*val138)+(val172*val139)+(val173*val140)+(val174*val141)+(val175*val142)+(val176*val124)+(val177*val125)+(val178*val126)+(val179*val127)+(val180*val128)+(val181*val129)+(val182*val130)+(val183*val131)+(val184*val132)+(val185*val64)+(val186*val69)+(val187*val74)+(val188*val78)+(val189*val83)+(val190*val88)+(val191*val93)+(val192*val98)+(val193*val103)+(val194*val107)+(val195*val111)+(val196*val115)+(val197*val119)+(val198*val148)+(val199*val152)+(val200*val155)+val156);
  data0_15360[alu11] = (val202+(val169*val5)+(val170*val9)+(val171*val12)+(val172*val15)+(val173*val18)+(val174*val21)+(val175*val24)+(val176*val28)+(val177*val32)+(val178*val36)+(val179*val40)+(val180*val44)+(val181*val48)+(val182*val52)+(val183*val56)+(val184*val60)+(val185*val65)+(val186*val70)+(val187*val75)+(val188*val79)+(val189*val84)+(val190*val89)+(val191*val94)+(val192*val99)+(val193*val104)+(val194*val108)+(val195*val112)+(val196*val116)+(val197*val120)+(val198*val149)+(val199*val153)+(val200*val158)+val159);
  data0_15360[alu12] = (val203+(val169*val6)+(val170*val161)+(val171*val162)+(val172*val163)+(val173*val164)+(val174*val165)+(val175*val25)+(val176*val29)+(val177*val33)+(val178*val37)+(val179*val41)+(val180*val45)+(val181*val49)+(val182*val53)+(val183*val57)+(val184*val61)+(val185*val66)+(val186*val71)+(val187*val76)+(val188*val80)+(val189*val85)+(val190*val90)+(val191*val95)+(val192*val100)+(val193*val143)+(val194*val144)+(val195*val145)+(val196*val146)+(val197*val147)+(val198*val150)+(val199*val154)+(val200*val166)+val167);
  data0_15360[alu13] = (val204+(val205*val4)+(val206*val7)+(val207*val10)+(val208*val13)+(val209*val16)+(val210*val19)+(val211*val22)+(val212*val26)+(val213*val30)+(val214*val34)+(val215*val38)+(val216*val42)+(val217*val46)+(val218*val50)+(val219*val54)+(val220*val58)+(val221*val62)+(val222*val67)+(val223*val72)+(val224*val77)+(val225*val81)+(val226*val86)+(val227*val91)+(val228*val96)+(val229*val101)+(val230*val105)+(val231*val109)+(val232*val113)+(val233*val117)+(val234*val121)+(val235*val151)+(val236*val133)+val134);
  data0_15360[alu14] = (val237+(val205*val136)+(val206*val137)+(val207*val138)+(val208*val139)+(val209*val140)+(val210*val141)+(val211*val142)+(val212*val124)+(val213*val125)+(val214*val126)+(val215*val127)+(val216*val128)+(val217*val129)+(val218*val130)+(val219*val131)+(val220*val132)+(val221*val64)+(val222*val69)+(val223*val74)+(val224*val78)+(val225*val83)+(val226*val88)+(val227*val93)+(val228*val98)+(val229*val103)+(val230*val107)+(val231*val111)+(val232*val115)+(val233*val119)+(val234*val148)+(val235*val152)+(val236*val155)+val156);
  data0_15360[alu15] = (val238+(val205*val5)+(val206*val9)+(val207*val12)+(val208*val15)+(val209*val18)+(val210*val21)+(val211*val24)+(val212*val28)+(val213*val32)+(val214*val36)+(val215*val40)+(val216*val44)+(val217*val48)+(val218*val52)+(val219*val56)+(val220*val60)+(val221*val65)+(val222*val70)+(val223*val75)+(val224*val79)+(val225*val84)+(val226*val89)+(val227*val94)+(val228*val99)+(val229*val104)+(val230*val108)+(val231*val112)+(val232*val116)+(val233*val120)+(val234*val149)+(val235*val153)+(val236*val158)+val159);
  data0_15360[alu16] = (val239+(val205*val6)+(val206*val161)+(val207*val162)+(val208*val163)+(val209*val164)+(val210*val165)+(val211*val25)+(val212*val29)+(val213*val33)+(val214*val37)+(val215*val41)+(val216*val45)+(val217*val49)+(val218*val53)+(val219*val57)+(val220*val61)+(val221*val66)+(val222*val71)+(val223*val76)+(val224*val80)+(val225*val85)+(val226*val90)+(val227*val95)+(val228*val100)+(val229*val143)+(val230*val144)+(val231*val145)+(val232*val146)+(val233*val147)+(val234*val150)+(val235*val154)+(val236*val166)+val167);
}`;

const r_120_10_16_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_1200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_15360:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1280:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var gidx1 = i32(gindex.y); /* 120 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val0 = data1_15360[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx1)<<7u)))];
    var val1 = data2_1280[(gidx0+(lidx0*80)+(Ridx0*10))];
    acc0[0] = (acc0[0]+(val0*val1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx103 = 0; Ridx103 < 16; Ridx103++) {
    var val2 = temp0[Ridx103];
    acc1[0] = (acc1[0]+val2);
  }
  var alu8 = ((bool(lidx0))!=true);
  if (alu8) {
    data0_1200[(gidx0+(gidx1*10))] = acc1[0];
  }
}`;

const r_15_8_10 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_120:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1200:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = ((gidx0*80)+(lidx0*10));
  var val0 = data1_1200[(alu0+1)];
  var val1 = data1_1200[(alu0+2)];
  var val2 = data1_1200[(alu0+3)];
  var val3 = data1_1200[(alu0+4)];
  var val4 = data1_1200[(alu0+5)];
  var val5 = data1_1200[(alu0+6)];
  var val6 = data1_1200[(alu0+7)];
  var val7 = data1_1200[(alu0+8)];
  var val8 = data1_1200[(alu0+9)];
  var val9 = data1_1200[alu0];
  var alu1 = select(val9,val0,(val9<val0));
  var alu2 = select(alu1,val1,(alu1<val1));
  var alu3 = select(alu2,val2,(alu2<val2));
  var alu4 = select(alu3,val3,(alu3<val3));
  var alu5 = select(alu4,val4,(alu4<val4));
  var alu6 = select(alu5,val5,(alu5<val5));
  var alu7 = select(alu6,val6,(alu6<val6));
  var alu8 = select(alu7,val7,(alu7<val7));
  var alu9 = select(alu8,val8,(alu8<val8));
  data0_120[(lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)))] = alu9;
}`;

const r_15_8_10n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_120:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1200:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_120:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 15 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = ((gidx0*80)+(lidx0*10));
  var val0 = data1_1200[(alu0+4)];
  var val1 = data1_1200[(alu0+5)];
  var val2 = data1_1200[(alu0+6)];
  var val3 = data1_1200[(alu0+7)];
  var val4 = data1_1200[(alu0+8)];
  var val5 = data1_1200[(alu0+9)];
  var val6 = data1_1200[alu0];
  var alu1 = (lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<3u)));
  var val7 = data2_120[alu1];
  var val8 = data1_1200[(alu0+1)];
  var val9 = data1_1200[(alu0+2)];
  var val10 = data1_1200[(alu0+3)];
  data0_120[alu1] = (log2((exp2(((val6-val7)*1.4426950408889634f))+exp2(((val8-val7)*1.4426950408889634f))+exp2(((val9-val7)*1.4426950408889634f))+exp2(((val10-val7)*1.4426950408889634f))+exp2(((val0-val7)*1.4426950408889634f))+exp2(((val1-val7)*1.4426950408889634f))+exp2(((val2-val7)*1.4426950408889634f))+exp2(((val3-val7)*1.4426950408889634f))+exp2(((val4-val7)*1.4426950408889634f))+exp2(((val5-val7)*1.4426950408889634f))))*0.6931471805599453f);
}`;

const E_5_5_8_2_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1200:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_120:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_120:array<f32>;
@compute @workgroup_size(8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 2 */
  var alu0 = (lidx1+bitcast<i32>((bitcast<u32>(gidx0)<<1u))+(gidx1*240)+(lidx0*30));
  var val0 = data1_1200[alu0];
  var alu1 = ((gidx1*24)+(lidx0*3));
  var alu2 = (alu1+2);
  var val1 = data2_120[alu2];
  var val2 = data3_120[alu1];
  var alu3 = (alu0+10);
  var val3 = data1_1200[alu3];
  var alu4 = (alu1+1);
  var val4 = data2_120[alu4];
  var val5 = data2_120[alu1];
  var val6 = data3_120[alu4];
  var alu5 = (alu0+20);
  var val7 = data1_1200[alu5];
  var val8 = data3_120[alu2];
  data0_1200[alu0] = ((val0-val5)-val2);
  data0_1200[alu3] = ((val3-val4)-val6);
  data0_1200[alu5] = ((val7-val1)-val8);
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 61440);;
    const input0 = createEmptyBuf(device, 480);;
    const buf_1 = createWeightBuf(device, 66560, getTensorBuffer(safetensor, metadata['embed']));
    const buf_2 = createEmptyBuf(device, 61440);;
    const buf_3 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.0.query.0']));
    const buf_4 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.query.1']));
    const buf_5 = createEmptyBuf(device, 61440);;
    const buf_6 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.0.key.0']));
    const buf_7 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.key.1']));
    const buf_8 = createEmptyBuf(device, 61440);;
    const buf_9 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.0.value.0']));
    const buf_10 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.value.1']));
    const buf_11 = createEmptyBuf(device, 230400);;
    const buf_12 = createEmptyBuf(device, 1920);;
    const buf_13 = createEmptyBuf(device, 1920);;
    const buf_14 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.0.out.0']));
    const buf_15 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.out.1']));
    const buf_16 = createEmptyBuf(device, 480);;
    const buf_17 = createEmptyBuf(device, 480);;
    const buf_18 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.ln1.0']));
    const buf_19 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.ln1.1']));
    const buf_20 = createEmptyBuf(device, 15360);;
    const buf_21 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.0.ff1.0']));
    const buf_22 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['tbs.0.ff1.1']));
    const buf_23 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.0.ff2.0']));
    const buf_24 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.ff2.1']));
    const buf_25 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.ln2.0']));
    const buf_26 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.0.ln2.1']));
    const buf_27 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.1.query.0']));
    const buf_28 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.query.1']));
    const buf_29 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.1.key.0']));
    const buf_30 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.key.1']));
    const buf_31 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.1.value.0']));
    const buf_32 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.value.1']));
    const buf_33 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.1.out.0']));
    const buf_34 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.out.1']));
    const buf_35 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.ln1.0']));
    const buf_36 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.ln1.1']));
    const buf_37 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.1.ff1.0']));
    const buf_38 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['tbs.1.ff1.1']));
    const buf_39 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.1.ff2.0']));
    const buf_40 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.ff2.1']));
    const buf_41 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.ln2.0']));
    const buf_42 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.1.ln2.1']));
    const buf_43 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.2.query.0']));
    const buf_44 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.query.1']));
    const buf_45 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.2.key.0']));
    const buf_46 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.key.1']));
    const buf_47 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.2.value.0']));
    const buf_48 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.value.1']));
    const buf_49 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.2.out.0']));
    const buf_50 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.out.1']));
    const buf_51 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.ln1.0']));
    const buf_52 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.ln1.1']));
    const buf_53 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.2.ff1.0']));
    const buf_54 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['tbs.2.ff1.1']));
    const buf_55 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.2.ff2.0']));
    const buf_56 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.ff2.1']));
    const buf_57 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.ln2.0']));
    const buf_58 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.2.ln2.1']));
    const buf_59 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.3.query.0']));
    const buf_60 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.query.1']));
    const buf_61 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.3.key.0']));
    const buf_62 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.key.1']));
    const buf_63 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.3.value.0']));
    const buf_64 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.value.1']));
    const buf_65 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['tbs.3.out.0']));
    const buf_66 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.out.1']));
    const buf_67 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.ln1.0']));
    const buf_68 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.ln1.1']));
    const buf_69 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.3.ff1.0']));
    const buf_70 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['tbs.3.ff1.1']));
    const buf_71 = createWeightBuf(device, 16384, getTensorBuffer(safetensor, metadata['tbs.3.ff2.0']));
    const buf_72 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.ff2.1']));
    const buf_73 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.ln2.0']));
    const buf_74 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['tbs.3.ln2.1']));
    const buf_75 = createEmptyBuf(device, 4800);;
    const buf_76 = createWeightBuf(device, 5120, getTensorBuffer(safetensor, metadata['final']));
    const output0 = createEmptyBuf(device, 4800);;

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_5_2_8_16_3_4_130, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_2_5_5_2_8_8_3_3_32, r_15_32_30_4, r_15_32_30_4n1, r_2_5_2_8_8_4_3_30_4, r_5_2_8_16_4_3_4_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_15_8_8_4_32_4, r_5_2_8_16_4_3_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_2_5_5_2_8_8_3_3_32, r_15_32_30_4, r_15_32_30_4n1, r_2_5_2_8_8_4_3_30_4, r_5_2_8_16_4_3_4_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_15_8_8_4_32_4, r_5_2_8_16_4_3_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_2_5_5_2_8_8_3_3_32, r_15_32_30_4, r_15_32_30_4n1, r_2_5_2_8_8_4_3_30_4, r_5_2_8_16_4_3_4_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_15_8_8_4_32_4, r_5_2_8_16_4_3_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_5_2_8_16_4_3_32_4, r_2_5_5_2_8_8_3_3_32, r_15_32_30_4, r_15_32_30_4n1, r_2_5_2_8_8_4_3_30_4, r_5_2_8_16_4_3_4_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_15_8_8_4_32_4, r_5_2_8_16_4_3_32, r_120_16_8, r_120_16_8n1, E_5_2_8_16_4_3, r_120_10_16_8, r_15_8_10, r_15_8_10n1, E_5_5_8_2_3];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_2, buf_0, buf_3, buf_4], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_5, buf_0, buf_6, buf_7], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_8, buf_0, buf_9, buf_10], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_11, buf_2, buf_5], [5, 5, 2]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_12, buf_11], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_13, buf_11, buf_12], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_5, buf_11, buf_12, buf_13, buf_8], [5, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_8, buf_0, buf_5, buf_14, buf_15], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_16, buf_8], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_17, buf_8, buf_16], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_5, buf_8, buf_16, buf_17, buf_18, buf_19], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_20, buf_5, buf_21, buf_22], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_8, buf_5, buf_20, buf_23, buf_24], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_17, buf_8], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_16, buf_8, buf_17], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_5, buf_8, buf_17, buf_16, buf_25, buf_26], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_8, buf_5, buf_27, buf_28], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_0, buf_5, buf_29, buf_30], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_2, buf_5, buf_31, buf_32], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_11, buf_8, buf_0], [5, 5, 2]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_13, buf_11], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_12, buf_11, buf_13], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_0, buf_11, buf_13, buf_12, buf_2], [5, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_2, buf_5, buf_0, buf_33, buf_34], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_16, buf_2], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_17, buf_2, buf_16], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_0, buf_2, buf_16, buf_17, buf_35, buf_36], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_20, buf_0, buf_37, buf_38], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_2, buf_0, buf_20, buf_39, buf_40], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_17, buf_2], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_16, buf_2, buf_17], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_0, buf_2, buf_17, buf_16, buf_41, buf_42], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_2, buf_0, buf_43, buf_44], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_5, buf_0, buf_45, buf_46], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_8, buf_0, buf_47, buf_48], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_11, buf_2, buf_5], [5, 5, 2]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_12, buf_11], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_13, buf_11, buf_12], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_5, buf_11, buf_12, buf_13, buf_8], [5, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_8, buf_0, buf_5, buf_49, buf_50], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_16, buf_8], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_17, buf_8, buf_16], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_5, buf_8, buf_16, buf_17, buf_51, buf_52], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_20, buf_5, buf_53, buf_54], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_8, buf_5, buf_20, buf_55, buf_56], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_17, buf_8], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_16, buf_8, buf_17], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_5, buf_8, buf_17, buf_16, buf_57, buf_58], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_8, buf_5, buf_59, buf_60], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_0, buf_5, buf_61, buf_62], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_2, buf_5, buf_63, buf_64], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_11, buf_8, buf_0], [5, 5, 2]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_13, buf_11], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_12, buf_11, buf_13], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_0, buf_11, buf_13, buf_12, buf_2], [5, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_2, buf_5, buf_0, buf_65, buf_66], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_16, buf_2], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_17, buf_2, buf_16], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_0, buf_2, buf_16, buf_17, buf_67, buf_68], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_20, buf_0, buf_69, buf_70], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_2, buf_0, buf_20, buf_71, buf_72], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_17, buf_2], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_16, buf_2, buf_17], [120, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_0, buf_2, buf_17, buf_16, buf_73, buf_74], [2, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_75, buf_0, buf_76], [10, 120, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_16, buf_75], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_17, buf_75, buf_16], [15, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [output0, buf_75, buf_16, buf_17], [5, 5, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default model;
