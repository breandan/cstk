
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

const E_53_2_4_16_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27136:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_212:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_12032:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_27136:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_212:array<i32>;
@group(0) @binding(6)var<storage,read_write>data5_512:array<f32>;
@compute @workgroup_size(4,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  var alu0 = (lidx0+bitcast<i32>((cast0<<2u)));
  var val0 = data1_212[alu0];
  var val1 = data4_212[alu0];
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu1 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu2 = (alu1+bitcast<i32>((bitcast<u32>(val0)<<7u)));
  var alu3 = ((-1<val0)&(val0<94));
  var val2 = select(0.0f, data2_12032[alu2], alu3);
  var alu4 = (alu1+bitcast<i32>((cast0<<9u))+bitcast<i32>((bitcast<u32>(lidx0)<<7u)));
  var val3 = data3_27136[alu4];
  var alu5 = (alu1+bitcast<i32>((bitcast<u32>(val1)<<7u)));
  var alu6 = ((-1<val1)&(val1<4));
  var val4 = select(0.0f, data5_512[alu5], alu6);
  var val5 = select(0.0f, data2_12032[(alu2+1)], alu3);
  var alu7 = (alu4+1);
  var val6 = data3_27136[alu7];
  var val7 = select(0.0f, data5_512[(alu5+1)], alu6);
  var val8 = select(0.0f, data2_12032[(alu2+2)], alu3);
  var alu8 = (alu4+2);
  var val9 = data3_27136[alu8];
  var val10 = select(0.0f, data5_512[(alu5+2)], alu6);
  var val11 = select(0.0f, data2_12032[(alu2+3)], alu3);
  var alu9 = (alu4+3);
  var val12 = data3_27136[alu9];
  var val13 = select(0.0f, data5_512[(alu5+3)], alu6);
  data0_27136[alu4] = (val2+val3+val4);
  data0_27136[alu7] = (val5+val6+val7);
  data0_27136[alu8] = (val8+val9+val10);
  data0_27136[alu9] = (val11+val12+val13);
}`;

const E_2_212_2_8_16_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1736704:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_13568:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2_12032:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_27136:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_13568:array<i32>;
@group(0) @binding(6)var<storage,read_write>data5_512:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 212 */
  var gidx2 = i32(gindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (gidx1+(gidx2*6784)+(lidx0*848));
  var val0 = data1_13568[alu0];
  var alu1 = (alu0+212);
  var val1 = data1_13568[alu1];
  var alu2 = (alu0+424);
  var val2 = data1_13568[alu2];
  var alu3 = (alu0+636);
  var val3 = data1_13568[alu3];
  var val4 = data4_13568[alu0];
  var val5 = data4_13568[alu1];
  var val6 = data4_13568[alu2];
  var val7 = data4_13568[alu3];
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu4 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu5 = (alu4+bitcast<i32>((bitcast<u32>(val0)<<7u)));
  var alu6 = ((-1<val0)&(val0<94));
  var val8 = select(0.0f, data2_12032[alu5], alu6);
  var alu7 = (alu4+bitcast<i32>((bitcast<u32>(gidx1)<<7u)));
  var val9 = data3_27136[alu7];
  var alu8 = (alu4+bitcast<i32>((bitcast<u32>(val4)<<7u)));
  var alu9 = ((-1<val4)&(val4<4));
  var val10 = select(0.0f, data5_512[alu8], alu9);
  var val11 = select(0.0f, data2_12032[(alu5+1)], alu6);
  var val12 = data3_27136[(alu7+1)];
  var val13 = select(0.0f, data5_512[(alu8+1)], alu9);
  var val14 = select(0.0f, data2_12032[(alu5+2)], alu6);
  var val15 = data3_27136[(alu7+2)];
  var val16 = select(0.0f, data5_512[(alu8+2)], alu9);
  var val17 = select(0.0f, data2_12032[(alu5+3)], alu6);
  var val18 = data3_27136[(alu7+3)];
  var alu10 = (alu4+bitcast<i32>((bitcast<u32>(val1)<<7u)));
  var alu11 = ((-1<val1)&(val1<94));
  var val19 = select(0.0f, data2_12032[alu10], alu11);
  var alu12 = (alu4+bitcast<i32>((bitcast<u32>(val5)<<7u)));
  var alu13 = ((-1<val5)&(val5<4));
  var val20 = select(0.0f, data5_512[alu12], alu13);
  var val21 = select(0.0f, data2_12032[(alu10+2)], alu11);
  var val22 = select(0.0f, data5_512[(alu12+2)], alu13);
  var val23 = select(0.0f, data2_12032[(alu10+3)], alu11);
  var val24 = select(0.0f, data5_512[(alu12+3)], alu13);
  var alu14 = (alu4+bitcast<i32>((bitcast<u32>(val2)<<7u)));
  var alu15 = ((-1<val2)&(val2<94));
  var val25 = select(0.0f, data2_12032[alu14], alu15);
  var alu16 = (alu4+bitcast<i32>((bitcast<u32>(val6)<<7u)));
  var alu17 = ((-1<val6)&(val6<4));
  var val26 = select(0.0f, data5_512[alu16], alu17);
  var val27 = select(0.0f, data5_512[(alu8+3)], alu9);
  var val28 = select(0.0f, data2_12032[(alu14+1)], alu15);
  var val29 = select(0.0f, data5_512[(alu16+1)], alu17);
  var alu18 = (alu4+bitcast<i32>((bitcast<u32>(val3)<<7u)));
  var alu19 = ((-1<val3)&(val3<94));
  var val30 = select(0.0f, data2_12032[alu18], alu19);
  var val31 = select(0.0f, data2_12032[(alu10+1)], alu11);
  var val32 = select(0.0f, data2_12032[(alu14+2)], alu15);
  var val33 = select(0.0f, data2_12032[(alu14+3)], alu15);
  var alu20 = (alu4+bitcast<i32>((bitcast<u32>(val7)<<7u)));
  var alu21 = ((-1<val7)&(val7<4));
  var val34 = select(0.0f, data5_512[alu20], alu21);
  var val35 = select(0.0f, data5_512[(alu12+1)], alu13);
  var val36 = select(0.0f, data5_512[(alu16+2)], alu17);
  var val37 = select(0.0f, data5_512[(alu16+3)], alu17);
  var val38 = select(0.0f, data2_12032[(alu18+1)], alu19);
  var val39 = select(0.0f, data5_512[(alu20+1)], alu21);
  var val40 = select(0.0f, data2_12032[(alu18+2)], alu19);
  var val41 = select(0.0f, data5_512[(alu20+2)], alu21);
  var val42 = select(0.0f, data2_12032[(alu18+3)], alu19);
  var val43 = select(0.0f, data5_512[(alu20+3)], alu21);
  var alu22 = (alu7+(gidx2*868352)+(lidx0*108544));
  data0_1736704[alu22] = (val8+val9+val10);
  data0_1736704[(alu22+1)] = (val11+val12+val13);
  data0_1736704[(alu22+2)] = (val14+val15+val16);
  data0_1736704[(alu22+3)] = (val17+val18+val27);
  data0_1736704[(alu22+27136)] = (val19+val9+val20);
  data0_1736704[(alu22+27137)] = (val31+val12+val35);
  data0_1736704[(alu22+27138)] = (val21+val15+val22);
  data0_1736704[(alu22+27139)] = (val23+val18+val24);
  data0_1736704[(alu22+54272)] = (val25+val9+val26);
  data0_1736704[(alu22+54273)] = (val28+val12+val29);
  data0_1736704[(alu22+54274)] = (val32+val15+val36);
  data0_1736704[(alu22+54275)] = (val33+val18+val37);
  data0_1736704[(alu22+81408)] = (val30+val9+val34);
  data0_1736704[(alu22+81409)] = (val38+val12+val39);
  data0_1736704[(alu22+81410)] = (val40+val15+val41);
  data0_1736704[(alu22+81411)] = (val42+val18+val43);
}`;

const r_53_8_16_3_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_81408:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_49152:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 16 */
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
    var cast0 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<9u))+cast0);
    var val0 = data1_27136[alu12];
    var alu13 = ((gidx0*6144)+(lidx0*384)+cast0);
    var val1 = data2_49152[(alu13+1)];
    var val2 = data2_49152[alu13];
    var val3 = data1_27136[(alu12+1)];
    var val4 = data1_27136[(alu12+2)];
    var val5 = data2_49152[(alu13+2)];
    var val6 = data1_27136[(alu12+3)];
    var val7 = data2_49152[(alu13+3)];
    var val8 = data1_27136[(alu12+128)];
    var val9 = data1_27136[(alu12+129)];
    var val10 = data1_27136[(alu12+130)];
    var val11 = data1_27136[(alu12+131)];
    var val12 = data1_27136[(alu12+256)];
    var val13 = data1_27136[(alu12+257)];
    var val14 = data1_27136[(alu12+258)];
    var val15 = data1_27136[(alu12+259)];
    var val16 = data1_27136[(alu12+384)];
    var val17 = data1_27136[(alu12+385)];
    var val18 = data1_27136[(alu12+386)];
    var val19 = data1_27136[(alu12+387)];
    var val20 = data2_49152[(alu13+128)];
    var val21 = data2_49152[(alu13+129)];
    var val22 = data2_49152[(alu13+130)];
    var val23 = data2_49152[(alu13+131)];
    var val24 = data2_49152[(alu13+256)];
    var val25 = data2_49152[(alu13+257)];
    var val26 = data2_49152[(alu13+258)];
    var val27 = data2_49152[(alu13+259)];
    acc0[0] = (acc0[0]+(val0*val2)+(val3*val1)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val2)+(val9*val1)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val2)+(val13*val1)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val2)+(val17*val1)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val3*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val3*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
  }
  var alu27 = ((gidx0*48)+(lidx0*3));
  var val28 = data3_384[(alu27+2)];
  var val29 = data3_384[alu27];
  var val30 = data3_384[(alu27+1)];
  var alu28 = (alu27+(gidx1*1536));
  data0_81408[(alu28+384)] = (acc0[1]+val29);
  data0_81408[(alu28+385)] = (acc0[5]+val30);
  data0_81408[(alu28+386)] = (acc0[9]+val28);
  data0_81408[(alu28+768)] = (acc0[2]+val29);
  data0_81408[(alu28+769)] = (acc0[6]+val30);
  data0_81408[(alu28+770)] = (acc0[10]+val28);
  data0_81408[(alu28+1152)] = (acc0[3]+val29);
  data0_81408[(alu28+1153)] = (acc0[7]+val30);
  data0_81408[(alu28+1154)] = (acc0[11]+val28);
  data0_81408[(alu28+1)] = (acc0[4]+val30);
  data0_81408[(alu28+2)] = (acc0[8]+val28);
  data0_81408[alu28] = (acc0[0]+val29);
}`;

const r_424_8_8_16_3_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_5210112:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_49152:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
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
    var cast0 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<12u))+bitcast<i32>((bitcast<u32>(lidx0)<<9u))+cast0);
    var val0 = data1_1736704[alu12];
    var alu13 = ((gidx0*6144)+(lidx1*384)+cast0);
    var val1 = data2_49152[(alu13+129)];
    var val2 = data2_49152[(alu13+130)];
    var val3 = data2_49152[(alu13+131)];
    var val4 = data2_49152[alu13];
    var val5 = data1_1736704[(alu12+1)];
    var val6 = data2_49152[(alu13+1)];
    var val7 = data1_1736704[(alu12+2)];
    var val8 = data2_49152[(alu13+2)];
    var val9 = data1_1736704[(alu12+3)];
    var val10 = data2_49152[(alu13+3)];
    var val11 = data1_1736704[(alu12+128)];
    var val12 = data1_1736704[(alu12+129)];
    var val13 = data1_1736704[(alu12+130)];
    var val14 = data1_1736704[(alu12+131)];
    var val15 = data1_1736704[(alu12+256)];
    var val16 = data1_1736704[(alu12+257)];
    var val17 = data1_1736704[(alu12+258)];
    var val18 = data1_1736704[(alu12+259)];
    var val19 = data1_1736704[(alu12+384)];
    var val20 = data1_1736704[(alu12+385)];
    var val21 = data1_1736704[(alu12+386)];
    var val22 = data1_1736704[(alu12+387)];
    var val23 = data2_49152[(alu13+128)];
    var val24 = data2_49152[(alu13+256)];
    var val25 = data2_49152[(alu13+257)];
    var val26 = data2_49152[(alu13+258)];
    var val27 = data2_49152[(alu13+259)];
    acc0[0] = (acc0[0]+(val0*val4)+(val5*val6)+(val7*val8)+(val9*val10));
    acc0[1] = (acc0[1]+(val11*val4)+(val12*val6)+(val13*val8)+(val14*val10));
    acc0[2] = (acc0[2]+(val15*val4)+(val16*val6)+(val17*val8)+(val18*val10));
    acc0[3] = (acc0[3]+(val19*val4)+(val20*val6)+(val21*val8)+(val22*val10));
    acc0[4] = (acc0[4]+(val0*val23)+(val5*val1)+(val7*val2)+(val9*val3));
    acc0[5] = (acc0[5]+(val11*val23)+(val12*val1)+(val13*val2)+(val14*val3));
    acc0[6] = (acc0[6]+(val15*val23)+(val16*val1)+(val17*val2)+(val18*val3));
    acc0[7] = (acc0[7]+(val19*val23)+(val20*val1)+(val21*val2)+(val22*val3));
    acc0[8] = (acc0[8]+(val0*val24)+(val5*val25)+(val7*val26)+(val9*val27));
    acc0[9] = (acc0[9]+(val11*val24)+(val12*val25)+(val13*val26)+(val14*val27));
    acc0[10] = (acc0[10]+(val15*val24)+(val16*val25)+(val17*val26)+(val18*val27));
    acc0[11] = (acc0[11]+(val19*val24)+(val20*val25)+(val21*val26)+(val22*val27));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_384[(alu27+2)];
  var val29 = data3_384[alu27];
  var val30 = data3_384[(alu27+1)];
  var alu28 = (alu27+(gidx1*12288)+(lidx0*1536));
  data0_5210112[(alu28+384)] = (acc0[1]+val29);
  data0_5210112[(alu28+385)] = (acc0[5]+val30);
  data0_5210112[(alu28+386)] = (acc0[9]+val28);
  data0_5210112[(alu28+768)] = (acc0[2]+val29);
  data0_5210112[(alu28+769)] = (acc0[6]+val30);
  data0_5210112[(alu28+770)] = (acc0[10]+val28);
  data0_5210112[(alu28+1152)] = (acc0[3]+val29);
  data0_5210112[(alu28+1153)] = (acc0[7]+val30);
  data0_5210112[(alu28+1154)] = (acc0[11]+val28);
  data0_5210112[(alu28+1)] = (acc0[4]+val30);
  data0_5210112[(alu28+2)] = (acc0[8]+val28);
  data0_5210112[alu28] = (acc0[0]+val29);
}`;

const r_53_53_8_4_4_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_359552:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_81408:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 53 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(lidx0)<<4u));
  var alu0 = ((gidx0*1536)+cast0);
  var val0 = data1_81408[(alu0+128)];
  var val1 = data1_81408[(alu0+129)];
  var val2 = data1_81408[(alu0+130)];
  var val3 = data1_81408[(alu0+131)];
  var val4 = data1_81408[(alu0+132)];
  var val5 = data1_81408[(alu0+133)];
  var val6 = data1_81408[(alu0+134)];
  var val7 = data1_81408[(alu0+135)];
  var val8 = data1_81408[(alu0+136)];
  var val9 = data1_81408[(alu0+137)];
  var val10 = data1_81408[(alu0+138)];
  var val11 = data1_81408[(alu0+139)];
  var val12 = data1_81408[(alu0+140)];
  var val13 = data1_81408[(alu0+141)];
  var val14 = data1_81408[(alu0+142)];
  var val15 = data1_81408[(alu0+143)];
  var val16 = data1_81408[(alu0+512)];
  var val17 = data1_81408[(alu0+513)];
  var val18 = data1_81408[(alu0+514)];
  var val19 = data1_81408[(alu0+515)];
  var val20 = data1_81408[(alu0+516)];
  var val21 = data1_81408[(alu0+517)];
  var val22 = data1_81408[(alu0+518)];
  var val23 = data1_81408[(alu0+519)];
  var val24 = data1_81408[(alu0+520)];
  var val25 = data1_81408[(alu0+521)];
  var val26 = data1_81408[(alu0+522)];
  var val27 = data1_81408[(alu0+523)];
  var val28 = data1_81408[(alu0+524)];
  var val29 = data1_81408[(alu0+525)];
  var val30 = data1_81408[(alu0+526)];
  var val31 = data1_81408[(alu0+527)];
  var val32 = data1_81408[(alu0+896)];
  var val33 = data1_81408[(alu0+897)];
  var val34 = data1_81408[(alu0+898)];
  var val35 = data1_81408[(alu0+899)];
  var val36 = data1_81408[(alu0+900)];
  var val37 = data1_81408[(alu0+901)];
  var val38 = data1_81408[(alu0+902)];
  var val39 = data1_81408[(alu0+903)];
  var val40 = data1_81408[(alu0+904)];
  var val41 = data1_81408[(alu0+905)];
  var val42 = data1_81408[(alu0+906)];
  var val43 = data1_81408[(alu0+907)];
  var val44 = data1_81408[(alu0+908)];
  var val45 = data1_81408[(alu0+909)];
  var val46 = data1_81408[(alu0+910)];
  var val47 = data1_81408[(alu0+911)];
  var val48 = data1_81408[(alu0+1280)];
  var val49 = data1_81408[(alu0+1281)];
  var val50 = data1_81408[(alu0+1282)];
  var val51 = data1_81408[(alu0+1283)];
  var val52 = data1_81408[(alu0+1284)];
  var val53 = data1_81408[(alu0+1285)];
  var val54 = data1_81408[(alu0+1286)];
  var val55 = data1_81408[(alu0+1287)];
  var val56 = data1_81408[(alu0+1288)];
  var val57 = data1_81408[(alu0+1289)];
  var val58 = data1_81408[(alu0+1290)];
  var val59 = data1_81408[(alu0+1291)];
  var val60 = data1_81408[(alu0+1292)];
  var val61 = data1_81408[(alu0+1293)];
  var val62 = data1_81408[(alu0+1294)];
  var val63 = data1_81408[(alu0+1295)];
  var gidx1 = i32(gindex.y); /* 53 */
  var alu1 = ((gidx1*1536)+cast0);
  var val64 = data1_81408[(alu1+1)];
  var val65 = data1_81408[(alu1+2)];
  var val66 = data1_81408[(alu1+3)];
  var val67 = data1_81408[(alu1+4)];
  var val68 = data1_81408[(alu1+5)];
  var val69 = data1_81408[(alu1+6)];
  var val70 = data1_81408[(alu1+7)];
  var val71 = data1_81408[(alu1+8)];
  var val72 = data1_81408[(alu1+9)];
  var val73 = data1_81408[(alu1+10)];
  var val74 = data1_81408[(alu1+11)];
  var val75 = data1_81408[(alu1+12)];
  var val76 = data1_81408[(alu1+13)];
  var val77 = data1_81408[(alu1+14)];
  var val78 = data1_81408[(alu1+15)];
  var val79 = data1_81408[(alu1+384)];
  var val80 = data1_81408[(alu1+385)];
  var val81 = data1_81408[(alu1+386)];
  var val82 = data1_81408[(alu1+387)];
  var val83 = data1_81408[(alu1+388)];
  var val84 = data1_81408[(alu1+389)];
  var val85 = data1_81408[(alu1+390)];
  var val86 = data1_81408[(alu1+391)];
  var val87 = data1_81408[(alu1+392)];
  var val88 = data1_81408[(alu1+393)];
  var val89 = data1_81408[(alu1+394)];
  var val90 = data1_81408[(alu1+395)];
  var val91 = data1_81408[(alu1+396)];
  var val92 = data1_81408[(alu1+397)];
  var val93 = data1_81408[(alu1+398)];
  var val94 = data1_81408[(alu1+399)];
  var val95 = data1_81408[(alu1+768)];
  var val96 = data1_81408[(alu1+769)];
  var val97 = data1_81408[(alu1+770)];
  var val98 = data1_81408[(alu1+771)];
  var val99 = data1_81408[(alu1+772)];
  var val100 = data1_81408[(alu1+773)];
  var val101 = data1_81408[(alu1+774)];
  var val102 = data1_81408[(alu1+775)];
  var val103 = data1_81408[(alu1+776)];
  var val104 = data1_81408[(alu1+777)];
  var val105 = data1_81408[(alu1+778)];
  var val106 = data1_81408[(alu1+779)];
  var val107 = data1_81408[(alu1+780)];
  var val108 = data1_81408[(alu1+781)];
  var val109 = data1_81408[(alu1+782)];
  var val110 = data1_81408[(alu1+783)];
  var val111 = data1_81408[(alu1+1152)];
  var val112 = data1_81408[(alu1+1153)];
  var val113 = data1_81408[(alu1+1154)];
  var val114 = data1_81408[(alu1+1155)];
  var val115 = data1_81408[(alu1+1156)];
  var val116 = data1_81408[(alu1+1157)];
  var val117 = data1_81408[(alu1+1158)];
  var val118 = data1_81408[(alu1+1159)];
  var val119 = data1_81408[(alu1+1160)];
  var val120 = data1_81408[(alu1+1161)];
  var val121 = data1_81408[(alu1+1162)];
  var val122 = data1_81408[(alu1+1163)];
  var val123 = data1_81408[(alu1+1164)];
  var val124 = data1_81408[(alu1+1165)];
  var val125 = data1_81408[(alu1+1166)];
  var val126 = data1_81408[(alu1+1167)];
  var val127 = data1_81408[alu1];
  var alu2 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+(gidx1*848)+(lidx0*44944));
  data0_359552[alu2] = (((val127*val0)+(val64*val1)+(val65*val2)+(val66*val3)+(val67*val4)+(val68*val5)+(val69*val6)+(val70*val7)+(val71*val8)+(val72*val9)+(val73*val10)+(val74*val11)+(val75*val12)+(val76*val13)+(val77*val14)+(val78*val15))*0.25f);
  data0_359552[(alu2+1)] = (((val127*val16)+(val64*val17)+(val65*val18)+(val66*val19)+(val67*val20)+(val68*val21)+(val69*val22)+(val70*val23)+(val71*val24)+(val72*val25)+(val73*val26)+(val74*val27)+(val75*val28)+(val76*val29)+(val77*val30)+(val78*val31))*0.25f);
  data0_359552[(alu2+2)] = (((val127*val32)+(val64*val33)+(val65*val34)+(val66*val35)+(val67*val36)+(val68*val37)+(val69*val38)+(val70*val39)+(val71*val40)+(val72*val41)+(val73*val42)+(val74*val43)+(val75*val44)+(val76*val45)+(val77*val46)+(val78*val47))*0.25f);
  data0_359552[(alu2+3)] = (((val127*val48)+(val64*val49)+(val65*val50)+(val66*val51)+(val67*val52)+(val68*val53)+(val69*val54)+(val70*val55)+(val71*val56)+(val72*val57)+(val73*val58)+(val74*val59)+(val75*val60)+(val76*val61)+(val77*val62)+(val78*val63))*0.25f);
  data0_359552[(alu2+212)] = (((val79*val0)+(val80*val1)+(val81*val2)+(val82*val3)+(val83*val4)+(val84*val5)+(val85*val6)+(val86*val7)+(val87*val8)+(val88*val9)+(val89*val10)+(val90*val11)+(val91*val12)+(val92*val13)+(val93*val14)+(val94*val15))*0.25f);
  data0_359552[(alu2+213)] = (((val79*val16)+(val80*val17)+(val81*val18)+(val82*val19)+(val83*val20)+(val84*val21)+(val85*val22)+(val86*val23)+(val87*val24)+(val88*val25)+(val89*val26)+(val90*val27)+(val91*val28)+(val92*val29)+(val93*val30)+(val94*val31))*0.25f);
  data0_359552[(alu2+214)] = (((val79*val32)+(val80*val33)+(val81*val34)+(val82*val35)+(val83*val36)+(val84*val37)+(val85*val38)+(val86*val39)+(val87*val40)+(val88*val41)+(val89*val42)+(val90*val43)+(val91*val44)+(val92*val45)+(val93*val46)+(val94*val47))*0.25f);
  data0_359552[(alu2+215)] = (((val79*val48)+(val80*val49)+(val81*val50)+(val82*val51)+(val83*val52)+(val84*val53)+(val85*val54)+(val86*val55)+(val87*val56)+(val88*val57)+(val89*val58)+(val90*val59)+(val91*val60)+(val92*val61)+(val93*val62)+(val94*val63))*0.25f);
  data0_359552[(alu2+424)] = (((val95*val0)+(val96*val1)+(val97*val2)+(val98*val3)+(val99*val4)+(val100*val5)+(val101*val6)+(val102*val7)+(val103*val8)+(val104*val9)+(val105*val10)+(val106*val11)+(val107*val12)+(val108*val13)+(val109*val14)+(val110*val15))*0.25f);
  data0_359552[(alu2+425)] = (((val95*val16)+(val96*val17)+(val97*val18)+(val98*val19)+(val99*val20)+(val100*val21)+(val101*val22)+(val102*val23)+(val103*val24)+(val104*val25)+(val105*val26)+(val106*val27)+(val107*val28)+(val108*val29)+(val109*val30)+(val110*val31))*0.25f);
  data0_359552[(alu2+426)] = (((val95*val32)+(val96*val33)+(val97*val34)+(val98*val35)+(val99*val36)+(val100*val37)+(val101*val38)+(val102*val39)+(val103*val40)+(val104*val41)+(val105*val42)+(val106*val43)+(val107*val44)+(val108*val45)+(val109*val46)+(val110*val47))*0.25f);
  data0_359552[(alu2+427)] = (((val95*val48)+(val96*val49)+(val97*val50)+(val98*val51)+(val99*val52)+(val100*val53)+(val101*val54)+(val102*val55)+(val103*val56)+(val104*val57)+(val105*val58)+(val106*val59)+(val107*val60)+(val108*val61)+(val109*val62)+(val110*val63))*0.25f);
  data0_359552[(alu2+636)] = (((val111*val0)+(val112*val1)+(val113*val2)+(val114*val3)+(val115*val4)+(val116*val5)+(val117*val6)+(val118*val7)+(val119*val8)+(val120*val9)+(val121*val10)+(val122*val11)+(val123*val12)+(val124*val13)+(val125*val14)+(val126*val15))*0.25f);
  data0_359552[(alu2+637)] = (((val111*val16)+(val112*val17)+(val113*val18)+(val114*val19)+(val115*val20)+(val116*val21)+(val117*val22)+(val118*val23)+(val119*val24)+(val120*val25)+(val121*val26)+(val122*val27)+(val123*val28)+(val124*val29)+(val125*val30)+(val126*val31))*0.25f);
  data0_359552[(alu2+638)] = (((val111*val32)+(val112*val33)+(val113*val34)+(val114*val35)+(val115*val36)+(val116*val37)+(val117*val38)+(val118*val39)+(val119*val40)+(val120*val41)+(val121*val42)+(val122*val43)+(val123*val44)+(val124*val45)+(val125*val46)+(val126*val47))*0.25f);
  data0_359552[(alu2+639)] = (((val111*val48)+(val112*val49)+(val113*val50)+(val114*val51)+(val115*val52)+(val116*val53)+(val117*val54)+(val118*val55)+(val119*val56)+(val120*val57)+(val121*val58)+(val122*val59)+(val123*val60)+(val124*val61)+(val125*val62)+(val126*val63))*0.25f);
}`;

const r_4_53_53_16_8_4_4_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_23011328:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_5210112:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 53 */
  var gidx2 = i32(gindex.z); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<i32>((bitcast<u32>(lidx1)<<4u));
  var alu0 = ((gidx2*1302528)+(lidx0*81408));
  var alu1 = ((gidx0*1536)+cast0+alu0);
  var val0 = data1_5210112[(alu1+128)];
  var val1 = data1_5210112[(alu1+129)];
  var val2 = data1_5210112[(alu1+130)];
  var val3 = data1_5210112[(alu1+131)];
  var val4 = data1_5210112[(alu1+132)];
  var val5 = data1_5210112[(alu1+133)];
  var val6 = data1_5210112[(alu1+134)];
  var val7 = data1_5210112[(alu1+135)];
  var val8 = data1_5210112[(alu1+136)];
  var val9 = data1_5210112[(alu1+137)];
  var val10 = data1_5210112[(alu1+138)];
  var val11 = data1_5210112[(alu1+139)];
  var val12 = data1_5210112[(alu1+140)];
  var val13 = data1_5210112[(alu1+141)];
  var val14 = data1_5210112[(alu1+142)];
  var val15 = data1_5210112[(alu1+143)];
  var val16 = data1_5210112[(alu1+512)];
  var val17 = data1_5210112[(alu1+513)];
  var val18 = data1_5210112[(alu1+514)];
  var val19 = data1_5210112[(alu1+515)];
  var val20 = data1_5210112[(alu1+516)];
  var val21 = data1_5210112[(alu1+517)];
  var val22 = data1_5210112[(alu1+518)];
  var val23 = data1_5210112[(alu1+519)];
  var val24 = data1_5210112[(alu1+520)];
  var val25 = data1_5210112[(alu1+521)];
  var val26 = data1_5210112[(alu1+522)];
  var val27 = data1_5210112[(alu1+523)];
  var val28 = data1_5210112[(alu1+524)];
  var val29 = data1_5210112[(alu1+525)];
  var val30 = data1_5210112[(alu1+526)];
  var val31 = data1_5210112[(alu1+527)];
  var val32 = data1_5210112[(alu1+896)];
  var val33 = data1_5210112[(alu1+897)];
  var val34 = data1_5210112[(alu1+898)];
  var val35 = data1_5210112[(alu1+899)];
  var val36 = data1_5210112[(alu1+900)];
  var val37 = data1_5210112[(alu1+901)];
  var val38 = data1_5210112[(alu1+902)];
  var val39 = data1_5210112[(alu1+903)];
  var val40 = data1_5210112[(alu1+904)];
  var val41 = data1_5210112[(alu1+905)];
  var val42 = data1_5210112[(alu1+906)];
  var val43 = data1_5210112[(alu1+907)];
  var val44 = data1_5210112[(alu1+908)];
  var val45 = data1_5210112[(alu1+909)];
  var val46 = data1_5210112[(alu1+910)];
  var val47 = data1_5210112[(alu1+911)];
  var val48 = data1_5210112[(alu1+1280)];
  var val49 = data1_5210112[(alu1+1281)];
  var val50 = data1_5210112[(alu1+1282)];
  var val51 = data1_5210112[(alu1+1283)];
  var val52 = data1_5210112[(alu1+1284)];
  var val53 = data1_5210112[(alu1+1285)];
  var val54 = data1_5210112[(alu1+1286)];
  var val55 = data1_5210112[(alu1+1287)];
  var val56 = data1_5210112[(alu1+1288)];
  var val57 = data1_5210112[(alu1+1289)];
  var val58 = data1_5210112[(alu1+1290)];
  var val59 = data1_5210112[(alu1+1291)];
  var val60 = data1_5210112[(alu1+1292)];
  var val61 = data1_5210112[(alu1+1293)];
  var val62 = data1_5210112[(alu1+1294)];
  var val63 = data1_5210112[(alu1+1295)];
  var gidx1 = i32(gindex.y); /* 53 */
  var alu2 = ((gidx1*1536)+cast0+alu0);
  var val64 = data1_5210112[(alu2+1)];
  var val65 = data1_5210112[(alu2+2)];
  var val66 = data1_5210112[(alu2+3)];
  var val67 = data1_5210112[(alu2+4)];
  var val68 = data1_5210112[(alu2+5)];
  var val69 = data1_5210112[(alu2+6)];
  var val70 = data1_5210112[(alu2+7)];
  var val71 = data1_5210112[(alu2+8)];
  var val72 = data1_5210112[(alu2+9)];
  var val73 = data1_5210112[(alu2+10)];
  var val74 = data1_5210112[(alu2+11)];
  var val75 = data1_5210112[(alu2+12)];
  var val76 = data1_5210112[(alu2+13)];
  var val77 = data1_5210112[(alu2+14)];
  var val78 = data1_5210112[(alu2+15)];
  var val79 = data1_5210112[(alu2+384)];
  var val80 = data1_5210112[(alu2+385)];
  var val81 = data1_5210112[(alu2+386)];
  var val82 = data1_5210112[(alu2+387)];
  var val83 = data1_5210112[(alu2+388)];
  var val84 = data1_5210112[(alu2+389)];
  var val85 = data1_5210112[(alu2+390)];
  var val86 = data1_5210112[(alu2+391)];
  var val87 = data1_5210112[(alu2+392)];
  var val88 = data1_5210112[(alu2+393)];
  var val89 = data1_5210112[(alu2+394)];
  var val90 = data1_5210112[(alu2+395)];
  var val91 = data1_5210112[(alu2+396)];
  var val92 = data1_5210112[(alu2+397)];
  var val93 = data1_5210112[(alu2+398)];
  var val94 = data1_5210112[(alu2+399)];
  var val95 = data1_5210112[(alu2+768)];
  var val96 = data1_5210112[(alu2+769)];
  var val97 = data1_5210112[(alu2+770)];
  var val98 = data1_5210112[(alu2+771)];
  var val99 = data1_5210112[(alu2+772)];
  var val100 = data1_5210112[(alu2+773)];
  var val101 = data1_5210112[(alu2+774)];
  var val102 = data1_5210112[(alu2+775)];
  var val103 = data1_5210112[(alu2+776)];
  var val104 = data1_5210112[(alu2+777)];
  var val105 = data1_5210112[(alu2+778)];
  var val106 = data1_5210112[(alu2+779)];
  var val107 = data1_5210112[(alu2+780)];
  var val108 = data1_5210112[(alu2+781)];
  var val109 = data1_5210112[(alu2+782)];
  var val110 = data1_5210112[(alu2+783)];
  var val111 = data1_5210112[(alu2+1152)];
  var val112 = data1_5210112[(alu2+1153)];
  var val113 = data1_5210112[(alu2+1154)];
  var val114 = data1_5210112[(alu2+1155)];
  var val115 = data1_5210112[(alu2+1156)];
  var val116 = data1_5210112[(alu2+1157)];
  var val117 = data1_5210112[(alu2+1158)];
  var val118 = data1_5210112[(alu2+1159)];
  var val119 = data1_5210112[(alu2+1160)];
  var val120 = data1_5210112[(alu2+1161)];
  var val121 = data1_5210112[(alu2+1162)];
  var val122 = data1_5210112[(alu2+1163)];
  var val123 = data1_5210112[(alu2+1164)];
  var val124 = data1_5210112[(alu2+1165)];
  var val125 = data1_5210112[(alu2+1166)];
  var val126 = data1_5210112[(alu2+1167)];
  var val127 = data1_5210112[alu2];
  var alu3 = (bitcast<i32>((bitcast<u32>(gidx0)<<2u))+(gidx1*848)+(lidx1*44944)+(gidx2*5752832)+(lidx0*359552));
  data0_23011328[alu3] = (((val127*val0)+(val64*val1)+(val65*val2)+(val66*val3)+(val67*val4)+(val68*val5)+(val69*val6)+(val70*val7)+(val71*val8)+(val72*val9)+(val73*val10)+(val74*val11)+(val75*val12)+(val76*val13)+(val77*val14)+(val78*val15))*0.25f);
  data0_23011328[(alu3+1)] = (((val127*val16)+(val64*val17)+(val65*val18)+(val66*val19)+(val67*val20)+(val68*val21)+(val69*val22)+(val70*val23)+(val71*val24)+(val72*val25)+(val73*val26)+(val74*val27)+(val75*val28)+(val76*val29)+(val77*val30)+(val78*val31))*0.25f);
  data0_23011328[(alu3+2)] = (((val127*val32)+(val64*val33)+(val65*val34)+(val66*val35)+(val67*val36)+(val68*val37)+(val69*val38)+(val70*val39)+(val71*val40)+(val72*val41)+(val73*val42)+(val74*val43)+(val75*val44)+(val76*val45)+(val77*val46)+(val78*val47))*0.25f);
  data0_23011328[(alu3+3)] = (((val127*val48)+(val64*val49)+(val65*val50)+(val66*val51)+(val67*val52)+(val68*val53)+(val69*val54)+(val70*val55)+(val71*val56)+(val72*val57)+(val73*val58)+(val74*val59)+(val75*val60)+(val76*val61)+(val77*val62)+(val78*val63))*0.25f);
  data0_23011328[(alu3+212)] = (((val79*val0)+(val80*val1)+(val81*val2)+(val82*val3)+(val83*val4)+(val84*val5)+(val85*val6)+(val86*val7)+(val87*val8)+(val88*val9)+(val89*val10)+(val90*val11)+(val91*val12)+(val92*val13)+(val93*val14)+(val94*val15))*0.25f);
  data0_23011328[(alu3+213)] = (((val79*val16)+(val80*val17)+(val81*val18)+(val82*val19)+(val83*val20)+(val84*val21)+(val85*val22)+(val86*val23)+(val87*val24)+(val88*val25)+(val89*val26)+(val90*val27)+(val91*val28)+(val92*val29)+(val93*val30)+(val94*val31))*0.25f);
  data0_23011328[(alu3+214)] = (((val79*val32)+(val80*val33)+(val81*val34)+(val82*val35)+(val83*val36)+(val84*val37)+(val85*val38)+(val86*val39)+(val87*val40)+(val88*val41)+(val89*val42)+(val90*val43)+(val91*val44)+(val92*val45)+(val93*val46)+(val94*val47))*0.25f);
  data0_23011328[(alu3+215)] = (((val79*val48)+(val80*val49)+(val81*val50)+(val82*val51)+(val83*val52)+(val84*val53)+(val85*val54)+(val86*val55)+(val87*val56)+(val88*val57)+(val89*val58)+(val90*val59)+(val91*val60)+(val92*val61)+(val93*val62)+(val94*val63))*0.25f);
  data0_23011328[(alu3+424)] = (((val95*val0)+(val96*val1)+(val97*val2)+(val98*val3)+(val99*val4)+(val100*val5)+(val101*val6)+(val102*val7)+(val103*val8)+(val104*val9)+(val105*val10)+(val106*val11)+(val107*val12)+(val108*val13)+(val109*val14)+(val110*val15))*0.25f);
  data0_23011328[(alu3+425)] = (((val95*val16)+(val96*val17)+(val97*val18)+(val98*val19)+(val99*val20)+(val100*val21)+(val101*val22)+(val102*val23)+(val103*val24)+(val104*val25)+(val105*val26)+(val106*val27)+(val107*val28)+(val108*val29)+(val109*val30)+(val110*val31))*0.25f);
  data0_23011328[(alu3+426)] = (((val95*val32)+(val96*val33)+(val97*val34)+(val98*val35)+(val99*val36)+(val100*val37)+(val101*val38)+(val102*val39)+(val103*val40)+(val104*val41)+(val105*val42)+(val106*val43)+(val107*val44)+(val108*val45)+(val109*val46)+(val110*val47))*0.25f);
  data0_23011328[(alu3+427)] = (((val95*val48)+(val96*val49)+(val97*val50)+(val98*val51)+(val99*val52)+(val100*val53)+(val101*val54)+(val102*val55)+(val103*val56)+(val104*val57)+(val105*val58)+(val106*val59)+(val107*val60)+(val108*val61)+(val109*val62)+(val110*val63))*0.25f);
  data0_23011328[(alu3+636)] = (((val111*val0)+(val112*val1)+(val113*val2)+(val114*val3)+(val115*val4)+(val116*val5)+(val117*val6)+(val118*val7)+(val119*val8)+(val120*val9)+(val121*val10)+(val122*val11)+(val123*val12)+(val124*val13)+(val125*val14)+(val126*val15))*0.25f);
  data0_23011328[(alu3+637)] = (((val111*val16)+(val112*val17)+(val113*val18)+(val114*val19)+(val115*val20)+(val116*val21)+(val117*val22)+(val118*val23)+(val119*val24)+(val120*val25)+(val121*val26)+(val122*val27)+(val123*val28)+(val124*val29)+(val125*val30)+(val126*val31))*0.25f);
  data0_23011328[(alu3+638)] = (((val111*val32)+(val112*val33)+(val113*val34)+(val114*val35)+(val115*val36)+(val116*val37)+(val117*val38)+(val118*val39)+(val119*val40)+(val120*val41)+(val121*val42)+(val122*val43)+(val123*val44)+(val124*val45)+(val125*val46)+(val126*val47))*0.25f);
  data0_23011328[(alu3+639)] = (((val111*val48)+(val112*val49)+(val113*val50)+(val114*val51)+(val115*val52)+(val116*val53)+(val117*val54)+(val118*val55)+(val119*val56)+(val120*val57)+(val121*val58)+(val122*val59)+(val123*val60)+(val124*val61)+(val125*val62)+(val126*val63))*0.25f);
}`;

const r_53_8_4_53_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1696:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_359552:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 53 */
  var lidx0 = i32(lindex.x); /* 8 */
  acc0[0] = (f32(-INFINITY));
  acc0[1] = (f32(-INFINITY));
  acc0[2] = (f32(-INFINITY));
  acc0[3] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu4 = ((gidx0*6784)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_359552[(alu4+1)];
    var val1 = data1_359552[(alu4+2)];
    var val2 = data1_359552[(alu4+3)];
    var val3 = data1_359552[(alu4+215)];
    var val4 = data1_359552[alu4];
    var val5 = data1_359552[(alu4+212)];
    var val6 = data1_359552[(alu4+213)];
    var val7 = data1_359552[(alu4+214)];
    var val8 = data1_359552[(alu4+424)];
    var val9 = data1_359552[(alu4+425)];
    var val10 = data1_359552[(alu4+426)];
    var val11 = data1_359552[(alu4+427)];
    var val12 = data1_359552[(alu4+636)];
    var val13 = data1_359552[(alu4+637)];
    var val14 = data1_359552[(alu4+638)];
    var val15 = data1_359552[(alu4+639)];
    var alu5 = select(acc0[0],val4,(acc0[0]<val4));
    var alu6 = select(acc0[1],val5,(acc0[1]<val5));
    var alu7 = select(acc0[2],val8,(acc0[2]<val8));
    var alu8 = select(acc0[3],val12,(acc0[3]<val12));
    var alu9 = select(alu5,val0,(alu5<val0));
    var alu10 = select(alu6,val6,(alu6<val6));
    var alu11 = select(alu7,val9,(alu7<val9));
    var alu12 = select(alu8,val13,(alu8<val13));
    var alu13 = select(alu9,val1,(alu9<val1));
    var alu14 = select(alu10,val7,(alu10<val7));
    var alu15 = select(alu11,val10,(alu11<val10));
    var alu16 = select(alu12,val14,(alu12<val14));
    var alu17 = select(alu13,val2,(alu13<val2));
    var alu18 = select(alu14,val3,(alu14<val3));
    var alu19 = select(alu15,val11,(alu15<val11));
    var alu20 = select(alu16,val15,(alu16<val15));
    acc0[0] = alu17;
    acc0[1] = alu18;
    acc0[2] = alu19;
    acc0[3] = alu20;
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<5u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  data0_1696[alu26] = acc0[0];
  data0_1696[(alu26+1)] = acc0[1];
  data0_1696[(alu26+2)] = acc0[2];
  data0_1696[(alu26+3)] = acc0[3];
}`;

const r_848_32_4_53_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_108544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 848 */
  var lidx0 = i32(lindex.x); /* 32 */
  acc0[0] = (f32(-INFINITY));
  acc0[1] = (f32(-INFINITY));
  acc0[2] = (f32(-INFINITY));
  acc0[3] = (f32(-INFINITY));
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu4 = ((gidx0*27136)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_23011328[(alu4+1)];
    var val1 = data1_23011328[(alu4+2)];
    var val2 = data1_23011328[(alu4+3)];
    var val3 = data1_23011328[(alu4+215)];
    var val4 = data1_23011328[alu4];
    var val5 = data1_23011328[(alu4+212)];
    var val6 = data1_23011328[(alu4+213)];
    var val7 = data1_23011328[(alu4+214)];
    var val8 = data1_23011328[(alu4+424)];
    var val9 = data1_23011328[(alu4+425)];
    var val10 = data1_23011328[(alu4+426)];
    var val11 = data1_23011328[(alu4+427)];
    var val12 = data1_23011328[(alu4+636)];
    var val13 = data1_23011328[(alu4+637)];
    var val14 = data1_23011328[(alu4+638)];
    var val15 = data1_23011328[(alu4+639)];
    var alu5 = select(acc0[0],val4,(acc0[0]<val4));
    var alu6 = select(acc0[1],val5,(acc0[1]<val5));
    var alu7 = select(acc0[2],val8,(acc0[2]<val8));
    var alu8 = select(acc0[3],val12,(acc0[3]<val12));
    var alu9 = select(alu5,val0,(alu5<val0));
    var alu10 = select(alu6,val6,(alu6<val6));
    var alu11 = select(alu7,val9,(alu7<val9));
    var alu12 = select(alu8,val13,(alu8<val13));
    var alu13 = select(alu9,val1,(alu9<val1));
    var alu14 = select(alu10,val7,(alu10<val7));
    var alu15 = select(alu11,val10,(alu11<val10));
    var alu16 = select(alu12,val14,(alu12<val14));
    var alu17 = select(alu13,val2,(alu13<val2));
    var alu18 = select(alu14,val3,(alu14<val3));
    var alu19 = select(alu15,val11,(alu15<val11));
    var alu20 = select(alu16,val15,(alu16<val15));
    acc0[0] = alu17;
    acc0[1] = alu18;
    acc0[2] = alu19;
    acc0[3] = alu20;
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  data0_108544[alu26] = acc0[0];
  data0_108544[(alu26+1)] = acc0[1];
  data0_108544[(alu26+2)] = acc0[2];
  data0_108544[(alu26+3)] = acc0[3];
}`;

const r_53_8_4_53_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1696:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_359552:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1696:array<f32>;
@compute @workgroup_size(8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 53 */
  var lidx0 = i32(lindex.x); /* 8 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<5u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var val0 = data2_1696[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_1696[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_1696[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_1696[alu3];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu8 = ((gidx0*6784)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val4 = data1_359552[(alu8+1)];
    var val5 = data1_359552[(alu8+2)];
    var val6 = data1_359552[(alu8+3)];
    var val7 = data1_359552[(alu8+212)];
    var val8 = data1_359552[(alu8+213)];
    var val9 = data1_359552[(alu8+214)];
    var val10 = data1_359552[(alu8+215)];
    var val11 = data1_359552[alu8];
    var val12 = data1_359552[(alu8+424)];
    var val13 = data1_359552[(alu8+425)];
    var val14 = data1_359552[(alu8+426)];
    var val15 = data1_359552[(alu8+427)];
    var val16 = data1_359552[(alu8+636)];
    var val17 = data1_359552[(alu8+637)];
    var val18 = data1_359552[(alu8+638)];
    var val19 = data1_359552[(alu8+639)];
    acc0[0] = (acc0[0]+exp2(((val11-val0)*1.4426950408889634f))+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val5-val0)*1.4426950408889634f))+exp2(((val6-val0)*1.4426950408889634f)));
    acc0[1] = (acc0[1]+exp2(((val7-val1)*1.4426950408889634f))+exp2(((val8-val1)*1.4426950408889634f))+exp2(((val9-val1)*1.4426950408889634f))+exp2(((val10-val1)*1.4426950408889634f)));
    acc0[2] = (acc0[2]+exp2(((val12-val2)*1.4426950408889634f))+exp2(((val13-val2)*1.4426950408889634f))+exp2(((val14-val2)*1.4426950408889634f))+exp2(((val15-val2)*1.4426950408889634f)));
    acc0[3] = (acc0[3]+exp2(((val16-val3)*1.4426950408889634f))+exp2(((val17-val3)*1.4426950408889634f))+exp2(((val18-val3)*1.4426950408889634f))+exp2(((val19-val3)*1.4426950408889634f)));
  }
  data0_1696[alu0] = acc0[0];
  data0_1696[alu1] = acc0[1];
  data0_1696[alu2] = acc0[2];
  data0_1696[alu3] = acc0[3];
}`;

const r_848_32_4_53_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_108544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_108544:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 848 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<7u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var val0 = data2_108544[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_108544[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_108544[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_108544[alu3];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu8 = ((gidx0*27136)+(lidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val4 = data1_23011328[(alu8+1)];
    var val5 = data1_23011328[(alu8+2)];
    var val6 = data1_23011328[(alu8+3)];
    var val7 = data1_23011328[(alu8+212)];
    var val8 = data1_23011328[(alu8+213)];
    var val9 = data1_23011328[(alu8+214)];
    var val10 = data1_23011328[(alu8+215)];
    var val11 = data1_23011328[alu8];
    var val12 = data1_23011328[(alu8+424)];
    var val13 = data1_23011328[(alu8+425)];
    var val14 = data1_23011328[(alu8+426)];
    var val15 = data1_23011328[(alu8+427)];
    var val16 = data1_23011328[(alu8+636)];
    var val17 = data1_23011328[(alu8+637)];
    var val18 = data1_23011328[(alu8+638)];
    var val19 = data1_23011328[(alu8+639)];
    acc0[0] = (acc0[0]+exp2(((val11-val0)*1.4426950408889634f))+exp2(((val4-val0)*1.4426950408889634f))+exp2(((val5-val0)*1.4426950408889634f))+exp2(((val6-val0)*1.4426950408889634f)));
    acc0[1] = (acc0[1]+exp2(((val7-val1)*1.4426950408889634f))+exp2(((val8-val1)*1.4426950408889634f))+exp2(((val9-val1)*1.4426950408889634f))+exp2(((val10-val1)*1.4426950408889634f)));
    acc0[2] = (acc0[2]+exp2(((val12-val2)*1.4426950408889634f))+exp2(((val13-val2)*1.4426950408889634f))+exp2(((val14-val2)*1.4426950408889634f))+exp2(((val15-val2)*1.4426950408889634f)));
    acc0[3] = (acc0[3]+exp2(((val16-val3)*1.4426950408889634f))+exp2(((val17-val3)*1.4426950408889634f))+exp2(((val18-val3)*1.4426950408889634f))+exp2(((val19-val3)*1.4426950408889634f)));
  }
  data0_108544[alu0] = acc0[0];
  data0_108544[alu1] = acc0[1];
  data0_108544[alu2] = acc0[2];
  data0_108544[alu3] = acc0[3];
}`;

const r_53_8_4_4_4_53_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27136:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_359552:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1696:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1696:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_81408:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 53 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((cast0<<2u))+(lidx0*212));
  var val0 = data2_1696[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_1696[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_1696[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_1696[alu3];
  var lidx1 = i32(lindex.y); /* 4 */
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu20 = ((gidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+(lidx0*44944));
    var val4 = data1_359552[(alu20+1)];
    var val5 = data1_359552[alu20];
    var alu21 = (bitcast<i32>((bitcast<u32>(lidx0)<<4u))+cast1+(Ridx0*1536));
    var val6 = data4_81408[(alu21+256)];
    var val7 = data4_81408[(alu21+640)];
    var val8 = data1_359552[(alu20+2)];
    var val9 = data4_81408[(alu21+1024)];
    var val10 = data1_359552[(alu20+3)];
    var val11 = data4_81408[(alu21+1408)];
    var val12 = data1_359552[(alu20+212)];
    var val13 = data1_359552[(alu20+213)];
    var val14 = data1_359552[(alu20+214)];
    var val15 = data1_359552[(alu20+215)];
    var val16 = data1_359552[(alu20+424)];
    var val17 = data1_359552[(alu20+425)];
    var val18 = data1_359552[(alu20+426)];
    var val19 = data1_359552[(alu20+427)];
    var val20 = data1_359552[(alu20+636)];
    var val21 = data1_359552[(alu20+637)];
    var val22 = data1_359552[(alu20+638)];
    var val23 = data1_359552[(alu20+639)];
    var val24 = data4_81408[(alu21+257)];
    var val25 = data4_81408[(alu21+641)];
    var val26 = data4_81408[(alu21+1025)];
    var val27 = data4_81408[(alu21+1409)];
    var val28 = data4_81408[(alu21+258)];
    var val29 = data4_81408[(alu21+642)];
    var val30 = data4_81408[(alu21+1026)];
    var val31 = data4_81408[(alu21+1410)];
    var val32 = data4_81408[(alu21+259)];
    var val33 = data4_81408[(alu21+643)];
    var val34 = data4_81408[(alu21+1027)];
    var val35 = data4_81408[(alu21+1411)];
    var alu22 = exp2(((val4-val0)*1.4426950408889634f));
    var alu23 = exp2(((val8-val0)*1.4426950408889634f));
    var alu24 = exp2(((val10-val0)*1.4426950408889634f));
    var alu25 = exp2(((val12-val1)*1.4426950408889634f));
    var alu26 = exp2(((val13-val1)*1.4426950408889634f));
    var alu27 = exp2(((val14-val1)*1.4426950408889634f));
    var alu28 = exp2(((val15-val1)*1.4426950408889634f));
    var alu29 = exp2(((val16-val2)*1.4426950408889634f));
    var alu30 = exp2(((val17-val2)*1.4426950408889634f));
    var alu31 = exp2(((val18-val2)*1.4426950408889634f));
    var alu32 = exp2(((val19-val2)*1.4426950408889634f));
    var alu33 = exp2(((val20-val3)*1.4426950408889634f));
    var alu34 = exp2(((val21-val3)*1.4426950408889634f));
    var alu35 = exp2(((val22-val3)*1.4426950408889634f));
    var alu36 = exp2(((val23-val3)*1.4426950408889634f));
    var alu37 = exp2(((val5-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu37*val6)+(alu22*val7)+(alu23*val9)+(alu24*val11));
    acc0[1] = (acc0[1]+(alu25*val6)+(alu26*val7)+(alu27*val9)+(alu28*val11));
    acc0[2] = (acc0[2]+(alu29*val6)+(alu30*val7)+(alu31*val9)+(alu32*val11));
    acc0[3] = (acc0[3]+(alu33*val6)+(alu34*val7)+(alu35*val9)+(alu36*val11));
    acc0[4] = (acc0[4]+(alu37*val24)+(alu22*val25)+(alu23*val26)+(alu24*val27));
    acc0[5] = (acc0[5]+(alu25*val24)+(alu26*val25)+(alu27*val26)+(alu28*val27));
    acc0[6] = (acc0[6]+(alu29*val24)+(alu30*val25)+(alu31*val26)+(alu32*val27));
    acc0[7] = (acc0[7]+(alu33*val24)+(alu34*val25)+(alu35*val26)+(alu36*val27));
    acc0[8] = (acc0[8]+(alu37*val28)+(alu22*val29)+(alu23*val30)+(alu24*val31));
    acc0[9] = (acc0[9]+(alu25*val28)+(alu26*val29)+(alu27*val30)+(alu28*val31));
    acc0[10] = (acc0[10]+(alu29*val28)+(alu30*val29)+(alu31*val30)+(alu32*val31));
    acc0[11] = (acc0[11]+(alu33*val28)+(alu34*val29)+(alu35*val30)+(alu36*val31));
    acc0[12] = (acc0[12]+(alu37*val32)+(alu22*val33)+(alu23*val34)+(alu24*val35));
    acc0[13] = (acc0[13]+(alu25*val32)+(alu26*val33)+(alu27*val34)+(alu28*val35));
    acc0[14] = (acc0[14]+(alu29*val32)+(alu30*val33)+(alu31*val34)+(alu32*val35));
    acc0[15] = (acc0[15]+(alu33*val32)+(alu34*val33)+(alu35*val34)+(alu36*val35));
  }
  var val36 = data3_1696[alu0];
  var val37 = data3_1696[alu1];
  var val38 = data3_1696[alu2];
  var val39 = data3_1696[alu3];
  var alu55 = (bitcast<i32>((cast0<<6u))+cast1+(lidx0*3392));
  var alu56 = (1/val36);
  data0_27136[alu55] = (acc0[0]*alu56);
  data0_27136[(alu55+1)] = (acc0[4]*alu56);
  data0_27136[(alu55+2)] = (acc0[8]*alu56);
  data0_27136[(alu55+3)] = (acc0[12]*alu56);
  var alu61 = (1/val37);
  data0_27136[(alu55+16)] = (acc0[1]*alu61);
  data0_27136[(alu55+17)] = (acc0[5]*alu61);
  data0_27136[(alu55+18)] = (acc0[9]*alu61);
  data0_27136[(alu55+19)] = (acc0[13]*alu61);
  var alu66 = (1/val38);
  data0_27136[(alu55+32)] = (acc0[2]*alu66);
  data0_27136[(alu55+33)] = (acc0[6]*alu66);
  data0_27136[(alu55+34)] = (acc0[10]*alu66);
  data0_27136[(alu55+35)] = (acc0[14]*alu66);
  var alu71 = (1/val39);
  data0_27136[(alu55+48)] = (acc0[3]*alu71);
  data0_27136[(alu55+49)] = (acc0[7]*alu71);
  data0_27136[(alu55+50)] = (acc0[11]*alu71);
  data0_27136[(alu55+51)] = (acc0[15]*alu71);
}`;

const r_16_53_4_8_4_4_4_53_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1736704:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_23011328:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_108544:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_108544:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_5210112:array<f32>;
@compute @workgroup_size(4,8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 53 */
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var cast0 = bitcast<u32>(gidx0);
  var alu0 = (bitcast<i32>((cast0<<2u))+(lidx1*212)+(gidx1*6784)+(lidx0*1696));
  var val0 = data2_108544[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_108544[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_108544[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_108544[alu3];
  var lidx2 = i32(lindex.z); /* 4 */
  var cast1 = bitcast<i32>((bitcast<u32>(lidx2)<<2u));
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 53; Ridx0++) {
    var alu20 = ((gidx0*848)+bitcast<i32>((bitcast<u32>(Ridx0)<<2u))+(lidx1*44944)+(gidx1*1438208)+(lidx0*359552));
    var val4 = data1_23011328[(alu20+2)];
    var val5 = data1_23011328[alu20];
    var alu21 = (bitcast<i32>((bitcast<u32>(lidx1)<<4u))+cast1+(Ridx0*1536)+(gidx1*325632)+(lidx0*81408));
    var val6 = data4_5210112[(alu21+256)];
    var val7 = data1_23011328[(alu20+1)];
    var val8 = data4_5210112[(alu21+257)];
    var val9 = data4_5210112[(alu21+640)];
    var val10 = data4_5210112[(alu21+641)];
    var val11 = data4_5210112[(alu21+1024)];
    var val12 = data1_23011328[(alu20+3)];
    var val13 = data4_5210112[(alu21+1025)];
    var val14 = data4_5210112[(alu21+1408)];
    var val15 = data1_23011328[(alu20+212)];
    var val16 = data1_23011328[(alu20+213)];
    var val17 = data1_23011328[(alu20+214)];
    var val18 = data1_23011328[(alu20+215)];
    var val19 = data1_23011328[(alu20+424)];
    var val20 = data1_23011328[(alu20+425)];
    var val21 = data1_23011328[(alu20+426)];
    var val22 = data1_23011328[(alu20+427)];
    var val23 = data1_23011328[(alu20+636)];
    var val24 = data1_23011328[(alu20+637)];
    var val25 = data1_23011328[(alu20+638)];
    var val26 = data1_23011328[(alu20+639)];
    var val27 = data4_5210112[(alu21+1409)];
    var val28 = data4_5210112[(alu21+258)];
    var val29 = data4_5210112[(alu21+1410)];
    var val30 = data4_5210112[(alu21+259)];
    var val31 = data4_5210112[(alu21+642)];
    var val32 = data4_5210112[(alu21+643)];
    var val33 = data4_5210112[(alu21+1026)];
    var val34 = data4_5210112[(alu21+1027)];
    var val35 = data4_5210112[(alu21+1411)];
    var alu22 = exp2(((val7-val0)*1.4426950408889634f));
    var alu23 = exp2(((val4-val0)*1.4426950408889634f));
    var alu24 = exp2(((val12-val0)*1.4426950408889634f));
    var alu25 = exp2(((val15-val1)*1.4426950408889634f));
    var alu26 = exp2(((val16-val1)*1.4426950408889634f));
    var alu27 = exp2(((val17-val1)*1.4426950408889634f));
    var alu28 = exp2(((val18-val1)*1.4426950408889634f));
    var alu29 = exp2(((val19-val2)*1.4426950408889634f));
    var alu30 = exp2(((val20-val2)*1.4426950408889634f));
    var alu31 = exp2(((val21-val2)*1.4426950408889634f));
    var alu32 = exp2(((val22-val2)*1.4426950408889634f));
    var alu33 = exp2(((val23-val3)*1.4426950408889634f));
    var alu34 = exp2(((val24-val3)*1.4426950408889634f));
    var alu35 = exp2(((val25-val3)*1.4426950408889634f));
    var alu36 = exp2(((val26-val3)*1.4426950408889634f));
    var alu37 = exp2(((val5-val0)*1.4426950408889634f));
    acc0[0] = (acc0[0]+(alu37*val6)+(alu22*val9)+(alu23*val11)+(alu24*val14));
    acc0[1] = (acc0[1]+(alu25*val6)+(alu26*val9)+(alu27*val11)+(alu28*val14));
    acc0[2] = (acc0[2]+(alu29*val6)+(alu30*val9)+(alu31*val11)+(alu32*val14));
    acc0[3] = (acc0[3]+(alu33*val6)+(alu34*val9)+(alu35*val11)+(alu36*val14));
    acc0[4] = (acc0[4]+(alu37*val8)+(alu22*val10)+(alu23*val13)+(alu24*val27));
    acc0[5] = (acc0[5]+(alu25*val8)+(alu26*val10)+(alu27*val13)+(alu28*val27));
    acc0[6] = (acc0[6]+(alu29*val8)+(alu30*val10)+(alu31*val13)+(alu32*val27));
    acc0[7] = (acc0[7]+(alu33*val8)+(alu34*val10)+(alu35*val13)+(alu36*val27));
    acc0[8] = (acc0[8]+(alu37*val28)+(alu22*val31)+(alu23*val33)+(alu24*val29));
    acc0[9] = (acc0[9]+(alu25*val28)+(alu26*val31)+(alu27*val33)+(alu28*val29));
    acc0[10] = (acc0[10]+(alu29*val28)+(alu30*val31)+(alu31*val33)+(alu32*val29));
    acc0[11] = (acc0[11]+(alu33*val28)+(alu34*val31)+(alu35*val33)+(alu36*val29));
    acc0[12] = (acc0[12]+(alu37*val30)+(alu22*val32)+(alu23*val34)+(alu24*val35));
    acc0[13] = (acc0[13]+(alu25*val30)+(alu26*val32)+(alu27*val34)+(alu28*val35));
    acc0[14] = (acc0[14]+(alu29*val30)+(alu30*val32)+(alu31*val34)+(alu32*val35));
    acc0[15] = (acc0[15]+(alu33*val30)+(alu34*val32)+(alu35*val34)+(alu36*val35));
  }
  var val36 = data3_108544[alu0];
  var val37 = data3_108544[alu1];
  var val38 = data3_108544[alu2];
  var val39 = data3_108544[alu3];
  var alu55 = (bitcast<i32>((cast0<<6u))+cast1+(lidx1*3392)+(gidx1*108544)+(lidx0*27136));
  var alu56 = (1/val36);
  data0_1736704[alu55] = (acc0[0]*alu56);
  data0_1736704[(alu55+1)] = (acc0[4]*alu56);
  data0_1736704[(alu55+2)] = (acc0[8]*alu56);
  data0_1736704[(alu55+3)] = (acc0[12]*alu56);
  var alu61 = (1/val37);
  data0_1736704[(alu55+16)] = (acc0[1]*alu61);
  data0_1736704[(alu55+17)] = (acc0[5]*alu61);
  data0_1736704[(alu55+18)] = (acc0[9]*alu61);
  data0_1736704[(alu55+19)] = (acc0[13]*alu61);
  var alu66 = (1/val38);
  data0_1736704[(alu55+32)] = (acc0[2]*alu66);
  data0_1736704[(alu55+33)] = (acc0[6]*alu66);
  data0_1736704[(alu55+34)] = (acc0[10]*alu66);
  data0_1736704[(alu55+35)] = (acc0[14]*alu66);
  var alu71 = (1/val39);
  data0_1736704[(alu55+48)] = (acc0[3]*alu71);
  data0_1736704[(alu55+49)] = (acc0[7]*alu71);
  data0_1736704[(alu55+50)] = (acc0[11]*alu71);
  data0_1736704[(alu55+51)] = (acc0[15]*alu71);
}`;

const r_53_2_16_4_4_8_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27136:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_27136:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_16384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0_0 = 0; Ridx0_0 < 8; Ridx0_0++) {
    var alu16 = (bitcast<i32>((cast1<<6u))+(Ridx0_0*3392));
    var val0 = data2_27136[alu16];
    var alu17 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast2<<9u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<4u)));
    var val1 = data3_16384[alu17];
    var val2 = data2_27136[(alu16+1)];
    var val3 = data3_16384[(alu17+1)];
    var val4 = data2_27136[(alu16+2)];
    var val5 = data3_16384[(alu17+2)];
    var val6 = data2_27136[(alu16+3)];
    var val7 = data3_16384[(alu17+3)];
    var val8 = data2_27136[(alu16+4)];
    var val9 = data3_16384[(alu17+4)];
    var val10 = data2_27136[(alu16+5)];
    var val11 = data3_16384[(alu17+5)];
    var val12 = data2_27136[(alu16+6)];
    var val13 = data3_16384[(alu17+6)];
    var val14 = data2_27136[(alu16+7)];
    var val15 = data3_16384[(alu17+7)];
    var val16 = data2_27136[(alu16+8)];
    var val17 = data3_16384[(alu17+8)];
    var val18 = data2_27136[(alu16+9)];
    var val19 = data3_16384[(alu17+9)];
    var val20 = data2_27136[(alu16+10)];
    var val21 = data3_16384[(alu17+10)];
    var val22 = data2_27136[(alu16+11)];
    var val23 = data3_16384[(alu17+11)];
    var val24 = data2_27136[(alu16+12)];
    var val25 = data3_16384[(alu17+12)];
    var val26 = data2_27136[(alu16+13)];
    var val27 = data3_16384[(alu17+13)];
    var val28 = data2_27136[(alu16+14)];
    var val29 = data3_16384[(alu17+14)];
    var val30 = data2_27136[(alu16+15)];
    var val31 = data3_16384[(alu17+15)];
    var val32 = data2_27136[(alu16+16)];
    var val33 = data2_27136[(alu16+17)];
    var val34 = data2_27136[(alu16+18)];
    var val35 = data2_27136[(alu16+19)];
    var val36 = data2_27136[(alu16+20)];
    var val37 = data2_27136[(alu16+21)];
    var val38 = data2_27136[(alu16+22)];
    var val39 = data2_27136[(alu16+23)];
    var val40 = data2_27136[(alu16+24)];
    var val41 = data2_27136[(alu16+25)];
    var val42 = data2_27136[(alu16+26)];
    var val43 = data2_27136[(alu16+27)];
    var val44 = data2_27136[(alu16+28)];
    var val45 = data2_27136[(alu16+29)];
    var val46 = data2_27136[(alu16+30)];
    var val47 = data2_27136[(alu16+31)];
    var val48 = data2_27136[(alu16+32)];
    var val49 = data2_27136[(alu16+33)];
    var val50 = data2_27136[(alu16+34)];
    var val51 = data2_27136[(alu16+35)];
    var val52 = data2_27136[(alu16+36)];
    var val53 = data2_27136[(alu16+37)];
    var val54 = data2_27136[(alu16+38)];
    var val55 = data2_27136[(alu16+39)];
    var val56 = data2_27136[(alu16+40)];
    var val57 = data2_27136[(alu16+41)];
    var val58 = data2_27136[(alu16+42)];
    var val59 = data2_27136[(alu16+43)];
    var val60 = data2_27136[(alu16+44)];
    var val61 = data2_27136[(alu16+45)];
    var val62 = data2_27136[(alu16+46)];
    var val63 = data2_27136[(alu16+47)];
    var val64 = data2_27136[(alu16+48)];
    var val65 = data2_27136[(alu16+49)];
    var val66 = data2_27136[(alu16+50)];
    var val67 = data2_27136[(alu16+51)];
    var val68 = data2_27136[(alu16+52)];
    var val69 = data2_27136[(alu16+53)];
    var val70 = data2_27136[(alu16+54)];
    var val71 = data2_27136[(alu16+55)];
    var val72 = data2_27136[(alu16+56)];
    var val73 = data2_27136[(alu16+57)];
    var val74 = data2_27136[(alu16+58)];
    var val75 = data2_27136[(alu16+59)];
    var val76 = data2_27136[(alu16+60)];
    var val77 = data2_27136[(alu16+61)];
    var val78 = data2_27136[(alu16+62)];
    var val79 = data2_27136[(alu16+63)];
    var val80 = data3_16384[(alu17+128)];
    var val81 = data3_16384[(alu17+129)];
    var val82 = data3_16384[(alu17+130)];
    var val83 = data3_16384[(alu17+131)];
    var val84 = data3_16384[(alu17+132)];
    var val85 = data3_16384[(alu17+133)];
    var val86 = data3_16384[(alu17+134)];
    var val87 = data3_16384[(alu17+135)];
    var val88 = data3_16384[(alu17+136)];
    var val89 = data3_16384[(alu17+137)];
    var val90 = data3_16384[(alu17+138)];
    var val91 = data3_16384[(alu17+139)];
    var val92 = data3_16384[(alu17+140)];
    var val93 = data3_16384[(alu17+141)];
    var val94 = data3_16384[(alu17+142)];
    var val95 = data3_16384[(alu17+143)];
    var val96 = data3_16384[(alu17+256)];
    var val97 = data3_16384[(alu17+257)];
    var val98 = data3_16384[(alu17+258)];
    var val99 = data3_16384[(alu17+259)];
    var val100 = data3_16384[(alu17+260)];
    var val101 = data3_16384[(alu17+261)];
    var val102 = data3_16384[(alu17+262)];
    var val103 = data3_16384[(alu17+263)];
    var val104 = data3_16384[(alu17+264)];
    var val105 = data3_16384[(alu17+265)];
    var val106 = data3_16384[(alu17+266)];
    var val107 = data3_16384[(alu17+267)];
    var val108 = data3_16384[(alu17+268)];
    var val109 = data3_16384[(alu17+269)];
    var val110 = data3_16384[(alu17+270)];
    var val111 = data3_16384[(alu17+271)];
    var val112 = data3_16384[(alu17+384)];
    var val113 = data3_16384[(alu17+385)];
    var val114 = data3_16384[(alu17+386)];
    var val115 = data3_16384[(alu17+387)];
    var val116 = data3_16384[(alu17+388)];
    var val117 = data3_16384[(alu17+389)];
    var val118 = data3_16384[(alu17+390)];
    var val119 = data3_16384[(alu17+391)];
    var val120 = data3_16384[(alu17+392)];
    var val121 = data3_16384[(alu17+393)];
    var val122 = data3_16384[(alu17+394)];
    var val123 = data3_16384[(alu17+395)];
    var val124 = data3_16384[(alu17+396)];
    var val125 = data3_16384[(alu17+397)];
    var val126 = data3_16384[(alu17+398)];
    var val127 = data3_16384[(alu17+399)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17)+(val18*val19)+(val20*val21)+(val22*val23)+(val24*val25)+(val26*val27)+(val28*val29)+(val30*val31));
    acc0[1] = (acc0[1]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7)+(val36*val9)+(val37*val11)+(val38*val13)+(val39*val15)+(val40*val17)+(val41*val19)+(val42*val21)+(val43*val23)+(val44*val25)+(val45*val27)+(val46*val29)+(val47*val31));
    acc0[2] = (acc0[2]+(val48*val1)+(val49*val3)+(val50*val5)+(val51*val7)+(val52*val9)+(val53*val11)+(val54*val13)+(val55*val15)+(val56*val17)+(val57*val19)+(val58*val21)+(val59*val23)+(val60*val25)+(val61*val27)+(val62*val29)+(val63*val31));
    acc0[3] = (acc0[3]+(val64*val1)+(val65*val3)+(val66*val5)+(val67*val7)+(val68*val9)+(val69*val11)+(val70*val13)+(val71*val15)+(val72*val17)+(val73*val19)+(val74*val21)+(val75*val23)+(val76*val25)+(val77*val27)+(val78*val29)+(val79*val31));
    acc0[4] = (acc0[4]+(val0*val80)+(val2*val81)+(val4*val82)+(val6*val83)+(val8*val84)+(val10*val85)+(val12*val86)+(val14*val87)+(val16*val88)+(val18*val89)+(val20*val90)+(val22*val91)+(val24*val92)+(val26*val93)+(val28*val94)+(val30*val95));
    acc0[5] = (acc0[5]+(val32*val80)+(val33*val81)+(val34*val82)+(val35*val83)+(val36*val84)+(val37*val85)+(val38*val86)+(val39*val87)+(val40*val88)+(val41*val89)+(val42*val90)+(val43*val91)+(val44*val92)+(val45*val93)+(val46*val94)+(val47*val95));
    acc0[6] = (acc0[6]+(val48*val80)+(val49*val81)+(val50*val82)+(val51*val83)+(val52*val84)+(val53*val85)+(val54*val86)+(val55*val87)+(val56*val88)+(val57*val89)+(val58*val90)+(val59*val91)+(val60*val92)+(val61*val93)+(val62*val94)+(val63*val95));
    acc0[7] = (acc0[7]+(val64*val80)+(val65*val81)+(val66*val82)+(val67*val83)+(val68*val84)+(val69*val85)+(val70*val86)+(val71*val87)+(val72*val88)+(val73*val89)+(val74*val90)+(val75*val91)+(val76*val92)+(val77*val93)+(val78*val94)+(val79*val95));
    acc0[8] = (acc0[8]+(val0*val96)+(val2*val97)+(val4*val98)+(val6*val99)+(val8*val100)+(val10*val101)+(val12*val102)+(val14*val103)+(val16*val104)+(val18*val105)+(val20*val106)+(val22*val107)+(val24*val108)+(val26*val109)+(val28*val110)+(val30*val111));
    acc0[9] = (acc0[9]+(val32*val96)+(val33*val97)+(val34*val98)+(val35*val99)+(val36*val100)+(val37*val101)+(val38*val102)+(val39*val103)+(val40*val104)+(val41*val105)+(val42*val106)+(val43*val107)+(val44*val108)+(val45*val109)+(val46*val110)+(val47*val111));
    acc0[10] = (acc0[10]+(val48*val96)+(val49*val97)+(val50*val98)+(val51*val99)+(val52*val100)+(val53*val101)+(val54*val102)+(val55*val103)+(val56*val104)+(val57*val105)+(val58*val106)+(val59*val107)+(val60*val108)+(val61*val109)+(val62*val110)+(val63*val111));
    acc0[11] = (acc0[11]+(val64*val96)+(val65*val97)+(val66*val98)+(val67*val99)+(val68*val100)+(val69*val101)+(val70*val102)+(val71*val103)+(val72*val104)+(val73*val105)+(val74*val106)+(val75*val107)+(val76*val108)+(val77*val109)+(val78*val110)+(val79*val111));
    acc0[12] = (acc0[12]+(val0*val112)+(val2*val113)+(val4*val114)+(val6*val115)+(val8*val116)+(val10*val117)+(val12*val118)+(val14*val119)+(val16*val120)+(val18*val121)+(val20*val122)+(val22*val123)+(val24*val124)+(val26*val125)+(val28*val126)+(val30*val127));
    acc0[13] = (acc0[13]+(val32*val112)+(val33*val113)+(val34*val114)+(val35*val115)+(val36*val116)+(val37*val117)+(val38*val118)+(val39*val119)+(val40*val120)+(val41*val121)+(val42*val122)+(val43*val123)+(val44*val124)+(val45*val125)+(val46*val126)+(val47*val127));
    acc0[14] = (acc0[14]+(val48*val112)+(val49*val113)+(val50*val114)+(val51*val115)+(val52*val116)+(val53*val117)+(val54*val118)+(val55*val119)+(val56*val120)+(val57*val121)+(val58*val122)+(val59*val123)+(val60*val124)+(val61*val125)+(val62*val126)+(val63*val127));
    acc0[15] = (acc0[15]+(val64*val112)+(val65*val113)+(val66*val114)+(val67*val115)+(val68*val116)+(val69*val117)+(val70*val118)+(val71*val119)+(val72*val120)+(val73*val121)+(val74*val122)+(val75*val123)+(val76*val124)+(val77*val125)+(val78*val126)+(val79*val127));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<9u)));
  var val128 = data1_27136[alu36];
  var val129 = data4_128[alu35];
  var alu37 = (alu36+1);
  var val130 = data1_27136[alu37];
  var val131 = data4_128[(alu35+1)];
  var alu38 = (alu36+2);
  var val132 = data1_27136[alu38];
  var val133 = data4_128[(alu35+2)];
  var alu39 = (alu36+3);
  var val134 = data1_27136[alu39];
  var val135 = data4_128[(alu35+3)];
  var alu40 = (alu36+128);
  var val136 = data1_27136[alu40];
  var alu41 = (alu36+129);
  var val137 = data1_27136[alu41];
  var alu42 = (alu36+130);
  var val138 = data1_27136[alu42];
  var alu43 = (alu36+131);
  var val139 = data1_27136[alu43];
  var alu44 = (alu36+256);
  var val140 = data1_27136[alu44];
  var alu45 = (alu36+257);
  var val141 = data1_27136[alu45];
  var alu46 = (alu36+258);
  var val142 = data1_27136[alu46];
  var alu47 = (alu36+259);
  var val143 = data1_27136[alu47];
  var alu48 = (alu36+384);
  var val144 = data1_27136[alu48];
  var alu49 = (alu36+385);
  var val145 = data1_27136[alu49];
  var alu50 = (alu36+386);
  var val146 = data1_27136[alu50];
  var alu51 = (alu36+387);
  var val147 = data1_27136[alu51];
  data0_27136[alu36] = (val128+acc0[0]+val129);
  data0_27136[alu37] = (val130+acc0[4]+val131);
  data0_27136[alu38] = (val132+acc0[8]+val133);
  data0_27136[alu39] = (val134+acc0[12]+val135);
  data0_27136[alu40] = (val136+acc0[1]+val129);
  data0_27136[alu41] = (val137+acc0[5]+val131);
  data0_27136[alu42] = (val138+acc0[9]+val133);
  data0_27136[alu43] = (val139+acc0[13]+val135);
  data0_27136[alu44] = (val140+acc0[2]+val129);
  data0_27136[alu45] = (val141+acc0[6]+val131);
  data0_27136[alu46] = (val142+acc0[10]+val133);
  data0_27136[alu47] = (val143+acc0[14]+val135);
  data0_27136[alu48] = (val144+acc0[3]+val129);
  data0_27136[alu49] = (val145+acc0[7]+val131);
  data0_27136[alu50] = (val146+acc0[11]+val133);
  data0_27136[alu51] = (val147+acc0[15]+val135);
}`;

const r_8_53_2_8_16_4_4_8_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1736704:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1736704:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_16384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 53 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx1);
  var alu0 = ((gidx2*217088)+(lidx0*27136));
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0_0 = 0; Ridx0_0 < 8; Ridx0_0++) {
    var alu17 = (bitcast<i32>((cast1<<6u))+(Ridx0_0*3392)+alu0);
    var val0 = data2_1736704[alu17];
    var alu18 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast2<<9u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<4u)));
    var val1 = data3_16384[alu18];
    var val2 = data2_1736704[(alu17+1)];
    var val3 = data3_16384[(alu18+1)];
    var val4 = data2_1736704[(alu17+2)];
    var val5 = data3_16384[(alu18+2)];
    var val6 = data2_1736704[(alu17+3)];
    var val7 = data3_16384[(alu18+3)];
    var val8 = data2_1736704[(alu17+4)];
    var val9 = data3_16384[(alu18+4)];
    var val10 = data2_1736704[(alu17+5)];
    var val11 = data3_16384[(alu18+5)];
    var val12 = data2_1736704[(alu17+6)];
    var val13 = data3_16384[(alu18+6)];
    var val14 = data2_1736704[(alu17+7)];
    var val15 = data3_16384[(alu18+7)];
    var val16 = data2_1736704[(alu17+8)];
    var val17 = data3_16384[(alu18+8)];
    var val18 = data2_1736704[(alu17+9)];
    var val19 = data3_16384[(alu18+9)];
    var val20 = data2_1736704[(alu17+10)];
    var val21 = data3_16384[(alu18+10)];
    var val22 = data2_1736704[(alu17+11)];
    var val23 = data3_16384[(alu18+11)];
    var val24 = data2_1736704[(alu17+12)];
    var val25 = data3_16384[(alu18+12)];
    var val26 = data2_1736704[(alu17+13)];
    var val27 = data3_16384[(alu18+13)];
    var val28 = data2_1736704[(alu17+14)];
    var val29 = data3_16384[(alu18+14)];
    var val30 = data2_1736704[(alu17+15)];
    var val31 = data3_16384[(alu18+15)];
    var val32 = data2_1736704[(alu17+16)];
    var val33 = data2_1736704[(alu17+17)];
    var val34 = data2_1736704[(alu17+18)];
    var val35 = data2_1736704[(alu17+19)];
    var val36 = data2_1736704[(alu17+20)];
    var val37 = data2_1736704[(alu17+21)];
    var val38 = data2_1736704[(alu17+22)];
    var val39 = data2_1736704[(alu17+23)];
    var val40 = data2_1736704[(alu17+24)];
    var val41 = data2_1736704[(alu17+25)];
    var val42 = data2_1736704[(alu17+26)];
    var val43 = data2_1736704[(alu17+27)];
    var val44 = data2_1736704[(alu17+28)];
    var val45 = data2_1736704[(alu17+29)];
    var val46 = data2_1736704[(alu17+30)];
    var val47 = data2_1736704[(alu17+31)];
    var val48 = data2_1736704[(alu17+32)];
    var val49 = data2_1736704[(alu17+33)];
    var val50 = data2_1736704[(alu17+34)];
    var val51 = data2_1736704[(alu17+35)];
    var val52 = data2_1736704[(alu17+36)];
    var val53 = data2_1736704[(alu17+37)];
    var val54 = data2_1736704[(alu17+38)];
    var val55 = data2_1736704[(alu17+39)];
    var val56 = data2_1736704[(alu17+40)];
    var val57 = data2_1736704[(alu17+41)];
    var val58 = data2_1736704[(alu17+42)];
    var val59 = data2_1736704[(alu17+43)];
    var val60 = data2_1736704[(alu17+44)];
    var val61 = data2_1736704[(alu17+45)];
    var val62 = data2_1736704[(alu17+46)];
    var val63 = data2_1736704[(alu17+47)];
    var val64 = data2_1736704[(alu17+48)];
    var val65 = data2_1736704[(alu17+49)];
    var val66 = data2_1736704[(alu17+50)];
    var val67 = data2_1736704[(alu17+51)];
    var val68 = data2_1736704[(alu17+52)];
    var val69 = data2_1736704[(alu17+53)];
    var val70 = data2_1736704[(alu17+54)];
    var val71 = data2_1736704[(alu17+55)];
    var val72 = data2_1736704[(alu17+56)];
    var val73 = data2_1736704[(alu17+57)];
    var val74 = data2_1736704[(alu17+58)];
    var val75 = data2_1736704[(alu17+59)];
    var val76 = data2_1736704[(alu17+60)];
    var val77 = data2_1736704[(alu17+61)];
    var val78 = data2_1736704[(alu17+62)];
    var val79 = data2_1736704[(alu17+63)];
    var val80 = data3_16384[(alu18+128)];
    var val81 = data3_16384[(alu18+129)];
    var val82 = data3_16384[(alu18+130)];
    var val83 = data3_16384[(alu18+131)];
    var val84 = data3_16384[(alu18+132)];
    var val85 = data3_16384[(alu18+133)];
    var val86 = data3_16384[(alu18+134)];
    var val87 = data3_16384[(alu18+135)];
    var val88 = data3_16384[(alu18+136)];
    var val89 = data3_16384[(alu18+137)];
    var val90 = data3_16384[(alu18+138)];
    var val91 = data3_16384[(alu18+139)];
    var val92 = data3_16384[(alu18+140)];
    var val93 = data3_16384[(alu18+141)];
    var val94 = data3_16384[(alu18+142)];
    var val95 = data3_16384[(alu18+143)];
    var val96 = data3_16384[(alu18+256)];
    var val97 = data3_16384[(alu18+257)];
    var val98 = data3_16384[(alu18+258)];
    var val99 = data3_16384[(alu18+259)];
    var val100 = data3_16384[(alu18+260)];
    var val101 = data3_16384[(alu18+261)];
    var val102 = data3_16384[(alu18+262)];
    var val103 = data3_16384[(alu18+263)];
    var val104 = data3_16384[(alu18+264)];
    var val105 = data3_16384[(alu18+265)];
    var val106 = data3_16384[(alu18+266)];
    var val107 = data3_16384[(alu18+267)];
    var val108 = data3_16384[(alu18+268)];
    var val109 = data3_16384[(alu18+269)];
    var val110 = data3_16384[(alu18+270)];
    var val111 = data3_16384[(alu18+271)];
    var val112 = data3_16384[(alu18+384)];
    var val113 = data3_16384[(alu18+385)];
    var val114 = data3_16384[(alu18+386)];
    var val115 = data3_16384[(alu18+387)];
    var val116 = data3_16384[(alu18+388)];
    var val117 = data3_16384[(alu18+389)];
    var val118 = data3_16384[(alu18+390)];
    var val119 = data3_16384[(alu18+391)];
    var val120 = data3_16384[(alu18+392)];
    var val121 = data3_16384[(alu18+393)];
    var val122 = data3_16384[(alu18+394)];
    var val123 = data3_16384[(alu18+395)];
    var val124 = data3_16384[(alu18+396)];
    var val125 = data3_16384[(alu18+397)];
    var val126 = data3_16384[(alu18+398)];
    var val127 = data3_16384[(alu18+399)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17)+(val18*val19)+(val20*val21)+(val22*val23)+(val24*val25)+(val26*val27)+(val28*val29)+(val30*val31));
    acc0[1] = (acc0[1]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7)+(val36*val9)+(val37*val11)+(val38*val13)+(val39*val15)+(val40*val17)+(val41*val19)+(val42*val21)+(val43*val23)+(val44*val25)+(val45*val27)+(val46*val29)+(val47*val31));
    acc0[2] = (acc0[2]+(val48*val1)+(val49*val3)+(val50*val5)+(val51*val7)+(val52*val9)+(val53*val11)+(val54*val13)+(val55*val15)+(val56*val17)+(val57*val19)+(val58*val21)+(val59*val23)+(val60*val25)+(val61*val27)+(val62*val29)+(val63*val31));
    acc0[3] = (acc0[3]+(val64*val1)+(val65*val3)+(val66*val5)+(val67*val7)+(val68*val9)+(val69*val11)+(val70*val13)+(val71*val15)+(val72*val17)+(val73*val19)+(val74*val21)+(val75*val23)+(val76*val25)+(val77*val27)+(val78*val29)+(val79*val31));
    acc0[4] = (acc0[4]+(val0*val80)+(val2*val81)+(val4*val82)+(val6*val83)+(val8*val84)+(val10*val85)+(val12*val86)+(val14*val87)+(val16*val88)+(val18*val89)+(val20*val90)+(val22*val91)+(val24*val92)+(val26*val93)+(val28*val94)+(val30*val95));
    acc0[5] = (acc0[5]+(val32*val80)+(val33*val81)+(val34*val82)+(val35*val83)+(val36*val84)+(val37*val85)+(val38*val86)+(val39*val87)+(val40*val88)+(val41*val89)+(val42*val90)+(val43*val91)+(val44*val92)+(val45*val93)+(val46*val94)+(val47*val95));
    acc0[6] = (acc0[6]+(val48*val80)+(val49*val81)+(val50*val82)+(val51*val83)+(val52*val84)+(val53*val85)+(val54*val86)+(val55*val87)+(val56*val88)+(val57*val89)+(val58*val90)+(val59*val91)+(val60*val92)+(val61*val93)+(val62*val94)+(val63*val95));
    acc0[7] = (acc0[7]+(val64*val80)+(val65*val81)+(val66*val82)+(val67*val83)+(val68*val84)+(val69*val85)+(val70*val86)+(val71*val87)+(val72*val88)+(val73*val89)+(val74*val90)+(val75*val91)+(val76*val92)+(val77*val93)+(val78*val94)+(val79*val95));
    acc0[8] = (acc0[8]+(val0*val96)+(val2*val97)+(val4*val98)+(val6*val99)+(val8*val100)+(val10*val101)+(val12*val102)+(val14*val103)+(val16*val104)+(val18*val105)+(val20*val106)+(val22*val107)+(val24*val108)+(val26*val109)+(val28*val110)+(val30*val111));
    acc0[9] = (acc0[9]+(val32*val96)+(val33*val97)+(val34*val98)+(val35*val99)+(val36*val100)+(val37*val101)+(val38*val102)+(val39*val103)+(val40*val104)+(val41*val105)+(val42*val106)+(val43*val107)+(val44*val108)+(val45*val109)+(val46*val110)+(val47*val111));
    acc0[10] = (acc0[10]+(val48*val96)+(val49*val97)+(val50*val98)+(val51*val99)+(val52*val100)+(val53*val101)+(val54*val102)+(val55*val103)+(val56*val104)+(val57*val105)+(val58*val106)+(val59*val107)+(val60*val108)+(val61*val109)+(val62*val110)+(val63*val111));
    acc0[11] = (acc0[11]+(val64*val96)+(val65*val97)+(val66*val98)+(val67*val99)+(val68*val100)+(val69*val101)+(val70*val102)+(val71*val103)+(val72*val104)+(val73*val105)+(val74*val106)+(val75*val107)+(val76*val108)+(val77*val109)+(val78*val110)+(val79*val111));
    acc0[12] = (acc0[12]+(val0*val112)+(val2*val113)+(val4*val114)+(val6*val115)+(val8*val116)+(val10*val117)+(val12*val118)+(val14*val119)+(val16*val120)+(val18*val121)+(val20*val122)+(val22*val123)+(val24*val124)+(val26*val125)+(val28*val126)+(val30*val127));
    acc0[13] = (acc0[13]+(val32*val112)+(val33*val113)+(val34*val114)+(val35*val115)+(val36*val116)+(val37*val117)+(val38*val118)+(val39*val119)+(val40*val120)+(val41*val121)+(val42*val122)+(val43*val123)+(val44*val124)+(val45*val125)+(val46*val126)+(val47*val127));
    acc0[14] = (acc0[14]+(val48*val112)+(val49*val113)+(val50*val114)+(val51*val115)+(val52*val116)+(val53*val117)+(val54*val118)+(val55*val119)+(val56*val120)+(val57*val121)+(val58*val122)+(val59*val123)+(val60*val124)+(val61*val125)+(val62*val126)+(val63*val127));
    acc0[15] = (acc0[15]+(val64*val112)+(val65*val113)+(val66*val114)+(val67*val115)+(val68*val116)+(val69*val117)+(val70*val118)+(val71*val119)+(val72*val120)+(val73*val121)+(val74*val122)+(val75*val123)+(val76*val124)+(val77*val125)+(val78*val126)+(val79*val127));
  }
  var alu36 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u)));
  var alu37 = (alu36+bitcast<i32>((cast1<<9u))+alu0);
  var val128 = data1_1736704[alu37];
  var val129 = data4_128[alu36];
  var alu38 = (alu37+1);
  var val130 = data1_1736704[alu38];
  var val131 = data4_128[(alu36+1)];
  var alu39 = (alu37+2);
  var val132 = data1_1736704[alu39];
  var val133 = data4_128[(alu36+2)];
  var alu40 = (alu37+3);
  var val134 = data1_1736704[alu40];
  var val135 = data4_128[(alu36+3)];
  var alu41 = (alu37+128);
  var val136 = data1_1736704[alu41];
  var alu42 = (alu37+129);
  var val137 = data1_1736704[alu42];
  var alu43 = (alu37+130);
  var val138 = data1_1736704[alu43];
  var alu44 = (alu37+131);
  var val139 = data1_1736704[alu44];
  var alu45 = (alu37+256);
  var val140 = data1_1736704[alu45];
  var alu46 = (alu37+257);
  var val141 = data1_1736704[alu46];
  var alu47 = (alu37+258);
  var val142 = data1_1736704[alu47];
  var alu48 = (alu37+259);
  var val143 = data1_1736704[alu48];
  var alu49 = (alu37+384);
  var val144 = data1_1736704[alu49];
  var alu50 = (alu37+385);
  var val145 = data1_1736704[alu50];
  var alu51 = (alu37+386);
  var val146 = data1_1736704[alu51];
  var alu52 = (alu37+387);
  var val147 = data1_1736704[alu52];
  data0_1736704[alu37] = (val128+acc0[0]+val129);
  data0_1736704[alu38] = (val130+acc0[4]+val131);
  data0_1736704[alu39] = (val132+acc0[8]+val133);
  data0_1736704[alu40] = (val134+acc0[12]+val135);
  data0_1736704[alu41] = (val136+acc0[1]+val129);
  data0_1736704[alu42] = (val137+acc0[5]+val131);
  data0_1736704[alu43] = (val138+acc0[9]+val133);
  data0_1736704[alu44] = (val139+acc0[13]+val135);
  data0_1736704[alu45] = (val140+acc0[2]+val129);
  data0_1736704[alu46] = (val141+acc0[6]+val131);
  data0_1736704[alu47] = (val142+acc0[10]+val133);
  data0_1736704[alu48] = (val143+acc0[14]+val135);
  data0_1736704[alu49] = (val144+acc0[3]+val129);
  data0_1736704[alu50] = (val145+acc0[7]+val131);
  data0_1736704[alu51] = (val146+acc0[11]+val133);
  data0_1736704[alu52] = (val147+acc0[15]+val135);
}`;

const r_212_16_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_212:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 212 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val0 = data1_27136[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
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
    data0_212[gidx0] = (acc1[0]*0.0078125f);
  }
}`;

const r_106_32_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_13568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 106 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu4 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<9u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val0 = data1_1736704[alu4];
    var val1 = data1_1736704[(alu4+1)];
    var val2 = data1_1736704[(alu4+2)];
    var val3 = data1_1736704[(alu4+3)];
    var val4 = data1_1736704[(alu4+128)];
    var val5 = data1_1736704[(alu4+129)];
    var val6 = data1_1736704[(alu4+130)];
    var val7 = data1_1736704[(alu4+131)];
    var val8 = data1_1736704[(alu4+256)];
    var val9 = data1_1736704[(alu4+257)];
    var val10 = data1_1736704[(alu4+258)];
    var val11 = data1_1736704[(alu4+259)];
    var val12 = data1_1736704[(alu4+384)];
    var val13 = data1_1736704[(alu4+385)];
    var val14 = data1_1736704[(alu4+386)];
    var val15 = data1_1736704[(alu4+387)];
    acc0[0] = (acc0[0]+val0+val1+val2+val3);
    acc0[1] = (acc0[1]+val4+val5+val6+val7);
    acc0[2] = (acc0[2]+val8+val9+val10+val11);
    acc0[3] = (acc0[3]+val12+val13+val14+val15);
  }
  var alu10 = (bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<2u)));
  data0_13568[alu10] = (acc0[0]*0.0078125f);
  data0_13568[(alu10+1)] = (acc0[1]*0.0078125f);
  data0_13568[(alu10+2)] = (acc0[2]*0.0078125f);
  data0_13568[(alu10+3)] = (acc0[3]*0.0078125f);
}`;

const r_212_16_8n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_212:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_212:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 212 */
  var val0 = data2_212[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val1 = data1_27136[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
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
    data0_212[gidx0] = (1/sqrt(((acc1[0]*0.0078125f)+1e-05f)));
  }
}`;

const r_106_32_4_32_4n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_13568:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_13568:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 106 */
  var lidx0 = i32(lindex.x); /* 32 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<7u))+bitcast<i32>((cast1<<2u)));
  var val0 = data2_13568[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_13568[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_13568[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_13568[alu3];
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu8 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<9u))+bitcast<i32>((bitcast<u32>(Ridx0)<<2u)));
    var val4 = data1_1736704[alu8];
    var val5 = data1_1736704[(alu8+1)];
    var val6 = data1_1736704[(alu8+2)];
    var val7 = data1_1736704[(alu8+3)];
    var val8 = data1_1736704[(alu8+128)];
    var val9 = data1_1736704[(alu8+129)];
    var val10 = data1_1736704[(alu8+130)];
    var val11 = data1_1736704[(alu8+131)];
    var val12 = data1_1736704[(alu8+256)];
    var val13 = data1_1736704[(alu8+257)];
    var val14 = data1_1736704[(alu8+258)];
    var val15 = data1_1736704[(alu8+259)];
    var val16 = data1_1736704[(alu8+384)];
    var val17 = data1_1736704[(alu8+385)];
    var val18 = data1_1736704[(alu8+386)];
    var val19 = data1_1736704[(alu8+387)];
    var alu9 = (val4-val0);
    var alu10 = (val8-val1);
    var alu11 = (val12-val2);
    var alu12 = (val16-val3);
    var alu13 = (val5-val0);
    var alu14 = (val9-val1);
    var alu15 = (val13-val2);
    var alu16 = (val17-val3);
    var alu17 = (val6-val0);
    var alu18 = (val10-val1);
    var alu19 = (val14-val2);
    var alu20 = (val18-val3);
    var alu21 = (val7-val0);
    var alu22 = (val11-val1);
    var alu23 = (val15-val2);
    var alu24 = (val19-val3);
    acc0[0] = (acc0[0]+(alu9*alu9)+(alu13*alu13)+(alu17*alu17)+(alu21*alu21));
    acc0[1] = (acc0[1]+(alu10*alu10)+(alu14*alu14)+(alu18*alu18)+(alu22*alu22));
    acc0[2] = (acc0[2]+(alu11*alu11)+(alu15*alu15)+(alu19*alu19)+(alu23*alu23));
    acc0[3] = (acc0[3]+(alu12*alu12)+(alu16*alu16)+(alu20*alu20)+(alu24*alu24));
  }
  data0_13568[alu0] = (1/sqrt(((acc0[0]*0.0078125f)+1e-05f)));
  data0_13568[alu1] = (1/sqrt(((acc0[1]*0.0078125f)+1e-05f)));
  data0_13568[alu2] = (1/sqrt(((acc0[2]*0.0078125f)+1e-05f)));
  data0_13568[alu3] = (1/sqrt(((acc0[3]*0.0078125f)+1e-05f)));
}`;

const E_53_2_16_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27136:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_212:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_212:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_128:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<9u)));
  var val0 = data1_27136[alu1];
  var cast1 = bitcast<i32>((cast0<<2u));
  var val1 = data2_212[cast1];
  var val2 = data3_212[cast1];
  var val3 = data4_128[alu0];
  var alu2 = (alu0+1);
  var val4 = data4_128[alu2];
  var val5 = data5_128[alu0];
  var alu3 = (alu1+1);
  var val6 = data1_27136[alu3];
  var val7 = data5_128[alu2];
  var alu4 = (alu1+2);
  var val8 = data1_27136[alu4];
  var alu5 = (alu0+2);
  var val9 = data4_128[alu5];
  var val10 = data5_128[alu5];
  var alu6 = (alu1+3);
  var val11 = data1_27136[alu6];
  var alu7 = (alu0+3);
  var val12 = data4_128[alu7];
  var val13 = data5_128[alu7];
  var alu8 = (alu1+128);
  var val14 = data1_27136[alu8];
  var alu9 = (cast1+1);
  var val15 = data2_212[alu9];
  var val16 = data3_212[alu9];
  var alu10 = (alu1+129);
  var val17 = data1_27136[alu10];
  var alu11 = (alu1+130);
  var val18 = data1_27136[alu11];
  var alu12 = (alu1+131);
  var val19 = data1_27136[alu12];
  var alu13 = (alu1+256);
  var val20 = data1_27136[alu13];
  var alu14 = (cast1+2);
  var val21 = data2_212[alu14];
  var val22 = data3_212[alu14];
  var alu15 = (alu1+257);
  var val23 = data1_27136[alu15];
  var alu16 = (alu1+258);
  var val24 = data1_27136[alu16];
  var alu17 = (alu1+259);
  var val25 = data1_27136[alu17];
  var alu18 = (alu1+384);
  var val26 = data1_27136[alu18];
  var alu19 = (cast1+3);
  var val27 = data2_212[alu19];
  var val28 = data3_212[alu19];
  var alu20 = (alu1+385);
  var val29 = data1_27136[alu20];
  var alu21 = (alu1+386);
  var val30 = data1_27136[alu21];
  var alu22 = (alu1+387);
  var val31 = data1_27136[alu22];
  data0_27136[alu1] = (((val0-val1)*val2*val3)+val5);
  data0_27136[alu3] = (((val6-val1)*val2*val4)+val7);
  data0_27136[alu4] = (((val8-val1)*val2*val9)+val10);
  data0_27136[alu6] = (((val11-val1)*val2*val12)+val13);
  data0_27136[alu8] = (((val14-val15)*val16*val3)+val5);
  data0_27136[alu10] = (((val17-val15)*val16*val4)+val7);
  data0_27136[alu11] = (((val18-val15)*val16*val9)+val10);
  data0_27136[alu12] = (((val19-val15)*val16*val12)+val13);
  data0_27136[alu13] = (((val20-val21)*val22*val3)+val5);
  data0_27136[alu15] = (((val23-val21)*val22*val4)+val7);
  data0_27136[alu16] = (((val24-val21)*val22*val9)+val10);
  data0_27136[alu17] = (((val25-val21)*val22*val12)+val13);
  data0_27136[alu18] = (((val26-val27)*val28*val3)+val5);
  data0_27136[alu20] = (((val29-val27)*val28*val4)+val7);
  data0_27136[alu21] = (((val30-val27)*val28*val9)+val10);
  data0_27136[alu22] = (((val31-val27)*val28*val12)+val13);
}`;

const E_424_2_8_16_4_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1736704:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_13568:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_13568:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<9u)));
  var val0 = data1_1736704[alu1];
  var alu2 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val1 = data2_13568[alu2];
  var val2 = data3_13568[alu2];
  var val3 = data4_128[alu0];
  var val4 = data5_128[alu0];
  var alu3 = (alu1+1);
  var val5 = data1_1736704[alu3];
  var alu4 = (alu0+1);
  var val6 = data4_128[alu4];
  var val7 = data5_128[alu4];
  var alu5 = (alu1+2);
  var val8 = data1_1736704[alu5];
  var alu6 = (alu0+2);
  var val9 = data4_128[alu6];
  var alu7 = (alu0+3);
  var val10 = data4_128[alu7];
  var val11 = data5_128[alu6];
  var alu8 = (alu1+3);
  var val12 = data1_1736704[alu8];
  var val13 = data5_128[alu7];
  var alu9 = (alu1+128);
  var val14 = data1_1736704[alu9];
  var alu10 = (alu2+1);
  var val15 = data2_13568[alu10];
  var val16 = data3_13568[alu10];
  var alu11 = (alu1+129);
  var val17 = data1_1736704[alu11];
  var alu12 = (alu1+130);
  var val18 = data1_1736704[alu12];
  var alu13 = (alu1+131);
  var val19 = data1_1736704[alu13];
  var alu14 = (alu1+256);
  var val20 = data1_1736704[alu14];
  var alu15 = (alu2+2);
  var val21 = data2_13568[alu15];
  var val22 = data3_13568[alu15];
  var alu16 = (alu1+257);
  var val23 = data1_1736704[alu16];
  var alu17 = (alu1+258);
  var val24 = data1_1736704[alu17];
  var alu18 = (alu1+259);
  var val25 = data1_1736704[alu18];
  var alu19 = (alu1+384);
  var val26 = data1_1736704[alu19];
  var alu20 = (alu2+3);
  var val27 = data2_13568[alu20];
  var val28 = data3_13568[alu20];
  var alu21 = (alu1+385);
  var val29 = data1_1736704[alu21];
  var alu22 = (alu1+386);
  var val30 = data1_1736704[alu22];
  var alu23 = (alu1+387);
  var val31 = data1_1736704[alu23];
  data0_1736704[alu1] = (((val0-val1)*val2*val3)+val4);
  data0_1736704[alu3] = (((val5-val1)*val2*val6)+val7);
  data0_1736704[alu5] = (((val8-val1)*val2*val9)+val11);
  data0_1736704[alu8] = (((val12-val1)*val2*val10)+val13);
  data0_1736704[alu9] = (((val14-val15)*val16*val3)+val4);
  data0_1736704[alu11] = (((val17-val15)*val16*val6)+val7);
  data0_1736704[alu12] = (((val18-val15)*val16*val9)+val11);
  data0_1736704[alu13] = (((val19-val15)*val16*val10)+val13);
  data0_1736704[alu14] = (((val20-val21)*val22*val3)+val4);
  data0_1736704[alu16] = (((val23-val21)*val22*val6)+val7);
  data0_1736704[alu17] = (((val24-val21)*val22*val9)+val11);
  data0_1736704[alu18] = (((val25-val21)*val22*val10)+val13);
  data0_1736704[alu19] = (((val26-val27)*val28*val3)+val4);
  data0_1736704[alu21] = (((val29-val27)*val28*val6)+val7);
  data0_1736704[alu22] = (((val30-val27)*val28*val9)+val11);
  data0_1736704[alu23] = (((val31-val27)*val28*val10)+val13);
}`;

const r_53_8_16_4_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_108544:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var cast3 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<9u))+cast3);
    var val0 = data1_27136[alu16];
    var alu17 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast2<<9u))+cast3);
    var val1 = data2_65536[alu17];
    var val2 = data1_27136[(alu16+1)];
    var val3 = data2_65536[(alu17+1)];
    var val4 = data1_27136[(alu16+2)];
    var val5 = data2_65536[(alu17+2)];
    var val6 = data1_27136[(alu16+3)];
    var val7 = data2_65536[(alu17+3)];
    var val8 = data1_27136[(alu16+128)];
    var val9 = data1_27136[(alu16+129)];
    var val10 = data1_27136[(alu16+130)];
    var val11 = data1_27136[(alu16+131)];
    var val12 = data1_27136[(alu16+256)];
    var val13 = data1_27136[(alu16+257)];
    var val14 = data1_27136[(alu16+258)];
    var val15 = data1_27136[(alu16+259)];
    var val16 = data1_27136[(alu16+384)];
    var val17 = data1_27136[(alu16+385)];
    var val18 = data1_27136[(alu16+386)];
    var val19 = data1_27136[(alu16+387)];
    var val20 = data2_65536[(alu17+128)];
    var val21 = data2_65536[(alu17+129)];
    var val22 = data2_65536[(alu17+130)];
    var val23 = data2_65536[(alu17+131)];
    var val24 = data2_65536[(alu17+256)];
    var val25 = data2_65536[(alu17+257)];
    var val26 = data2_65536[(alu17+258)];
    var val27 = data2_65536[(alu17+259)];
    var val28 = data2_65536[(alu17+384)];
    var val29 = data2_65536[(alu17+385)];
    var val30 = data2_65536[(alu17+386)];
    var val31 = data2_65536[(alu17+387)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u)));
  var val32 = data3_512[alu35];
  var val33 = data3_512[(alu35+1)];
  var val34 = data3_512[(alu35+2)];
  var val35 = data3_512[(alu35+3)];
  var alu36 = (alu35+bitcast<i32>((cast1<<11u)));
  var alu37 = (acc0[0]+val32);
  var alu38 = (acc0[4]+val33);
  var alu39 = (acc0[8]+val34);
  var alu40 = (acc0[12]+val35);
  data0_108544[alu36] = ((1/(1.0f+exp2(((alu37+(0.044715f*alu37*alu37*alu37))*-2.302208198144325f))))*alu37);
  data0_108544[(alu36+1)] = ((1/(1.0f+exp2(((alu38+(0.044715f*alu38*alu38*alu38))*-2.302208198144325f))))*alu38);
  data0_108544[(alu36+2)] = ((1/(1.0f+exp2(((alu39+(0.044715f*alu39*alu39*alu39))*-2.302208198144325f))))*alu39);
  data0_108544[(alu36+3)] = ((1/(1.0f+exp2(((alu40+(0.044715f*alu40*alu40*alu40))*-2.302208198144325f))))*alu40);
  var alu45 = (acc0[1]+val32);
  var alu46 = (acc0[5]+val33);
  var alu47 = (acc0[9]+val34);
  var alu48 = (acc0[13]+val35);
  data0_108544[(alu36+512)] = ((1/(1.0f+exp2(((alu45+(0.044715f*alu45*alu45*alu45))*-2.302208198144325f))))*alu45);
  data0_108544[(alu36+513)] = ((1/(1.0f+exp2(((alu46+(0.044715f*alu46*alu46*alu46))*-2.302208198144325f))))*alu46);
  data0_108544[(alu36+514)] = ((1/(1.0f+exp2(((alu47+(0.044715f*alu47*alu47*alu47))*-2.302208198144325f))))*alu47);
  data0_108544[(alu36+515)] = ((1/(1.0f+exp2(((alu48+(0.044715f*alu48*alu48*alu48))*-2.302208198144325f))))*alu48);
  var alu53 = (acc0[2]+val32);
  var alu54 = (acc0[6]+val33);
  var alu55 = (acc0[10]+val34);
  var alu56 = (acc0[14]+val35);
  data0_108544[(alu36+1024)] = ((1/(1.0f+exp2(((alu53+(0.044715f*alu53*alu53*alu53))*-2.302208198144325f))))*alu53);
  data0_108544[(alu36+1025)] = ((1/(1.0f+exp2(((alu54+(0.044715f*alu54*alu54*alu54))*-2.302208198144325f))))*alu54);
  data0_108544[(alu36+1026)] = ((1/(1.0f+exp2(((alu55+(0.044715f*alu55*alu55*alu55))*-2.302208198144325f))))*alu55);
  data0_108544[(alu36+1027)] = ((1/(1.0f+exp2(((alu56+(0.044715f*alu56*alu56*alu56))*-2.302208198144325f))))*alu56);
  var alu61 = (acc0[3]+val32);
  var alu62 = (acc0[7]+val33);
  var alu63 = (acc0[11]+val34);
  var alu64 = (acc0[15]+val35);
  data0_108544[(alu36+1536)] = ((1/(1.0f+exp2(((alu61+(0.044715f*alu61*alu61*alu61))*-2.302208198144325f))))*alu61);
  data0_108544[(alu36+1537)] = ((1/(1.0f+exp2(((alu62+(0.044715f*alu62*alu62*alu62))*-2.302208198144325f))))*alu62);
  data0_108544[(alu36+1538)] = ((1/(1.0f+exp2(((alu63+(0.044715f*alu63*alu63*alu63))*-2.302208198144325f))))*alu63);
  data0_108544[(alu36+1539)] = ((1/(1.0f+exp2(((alu64+(0.044715f*alu64*alu64*alu64))*-2.302208198144325f))))*alu64);
}`;

const r_424_8_8_16_4_4_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_6946816:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var cast4 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<12u))+bitcast<i32>((cast2<<9u))+cast4);
    var val0 = data1_1736704[alu16];
    var alu17 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<9u))+cast4);
    var val1 = data2_65536[alu17];
    var val2 = data1_1736704[(alu16+1)];
    var val3 = data2_65536[(alu17+1)];
    var val4 = data1_1736704[(alu16+2)];
    var val5 = data2_65536[(alu17+2)];
    var val6 = data1_1736704[(alu16+3)];
    var val7 = data2_65536[(alu17+3)];
    var val8 = data1_1736704[(alu16+128)];
    var val9 = data1_1736704[(alu16+129)];
    var val10 = data1_1736704[(alu16+130)];
    var val11 = data1_1736704[(alu16+131)];
    var val12 = data1_1736704[(alu16+256)];
    var val13 = data1_1736704[(alu16+257)];
    var val14 = data1_1736704[(alu16+258)];
    var val15 = data1_1736704[(alu16+259)];
    var val16 = data1_1736704[(alu16+384)];
    var val17 = data1_1736704[(alu16+385)];
    var val18 = data1_1736704[(alu16+386)];
    var val19 = data1_1736704[(alu16+387)];
    var val20 = data2_65536[(alu17+128)];
    var val21 = data2_65536[(alu17+129)];
    var val22 = data2_65536[(alu17+130)];
    var val23 = data2_65536[(alu17+131)];
    var val24 = data2_65536[(alu17+256)];
    var val25 = data2_65536[(alu17+257)];
    var val26 = data2_65536[(alu17+258)];
    var val27 = data2_65536[(alu17+259)];
    var val28 = data2_65536[(alu17+384)];
    var val29 = data2_65536[(alu17+385)];
    var val30 = data2_65536[(alu17+386)];
    var val31 = data2_65536[(alu17+387)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var val32 = data3_512[alu35];
  var val33 = data3_512[(alu35+1)];
  var val34 = data3_512[(alu35+2)];
  var val35 = data3_512[(alu35+3)];
  var alu36 = (alu35+bitcast<i32>((cast1<<14u))+bitcast<i32>((cast2<<11u)));
  var alu37 = (acc0[0]+val32);
  var alu38 = (acc0[4]+val33);
  var alu39 = (acc0[8]+val34);
  var alu40 = (acc0[12]+val35);
  data0_6946816[alu36] = ((1/(1.0f+exp2(((alu37+(0.044715f*alu37*alu37*alu37))*-2.302208198144325f))))*alu37);
  data0_6946816[(alu36+1)] = ((1/(1.0f+exp2(((alu38+(0.044715f*alu38*alu38*alu38))*-2.302208198144325f))))*alu38);
  data0_6946816[(alu36+2)] = ((1/(1.0f+exp2(((alu39+(0.044715f*alu39*alu39*alu39))*-2.302208198144325f))))*alu39);
  data0_6946816[(alu36+3)] = ((1/(1.0f+exp2(((alu40+(0.044715f*alu40*alu40*alu40))*-2.302208198144325f))))*alu40);
  var alu45 = (acc0[1]+val32);
  var alu46 = (acc0[5]+val33);
  var alu47 = (acc0[9]+val34);
  var alu48 = (acc0[13]+val35);
  data0_6946816[(alu36+512)] = ((1/(1.0f+exp2(((alu45+(0.044715f*alu45*alu45*alu45))*-2.302208198144325f))))*alu45);
  data0_6946816[(alu36+513)] = ((1/(1.0f+exp2(((alu46+(0.044715f*alu46*alu46*alu46))*-2.302208198144325f))))*alu46);
  data0_6946816[(alu36+514)] = ((1/(1.0f+exp2(((alu47+(0.044715f*alu47*alu47*alu47))*-2.302208198144325f))))*alu47);
  data0_6946816[(alu36+515)] = ((1/(1.0f+exp2(((alu48+(0.044715f*alu48*alu48*alu48))*-2.302208198144325f))))*alu48);
  var alu53 = (acc0[2]+val32);
  var alu54 = (acc0[6]+val33);
  var alu55 = (acc0[10]+val34);
  var alu56 = (acc0[14]+val35);
  data0_6946816[(alu36+1024)] = ((1/(1.0f+exp2(((alu53+(0.044715f*alu53*alu53*alu53))*-2.302208198144325f))))*alu53);
  data0_6946816[(alu36+1025)] = ((1/(1.0f+exp2(((alu54+(0.044715f*alu54*alu54*alu54))*-2.302208198144325f))))*alu54);
  data0_6946816[(alu36+1026)] = ((1/(1.0f+exp2(((alu55+(0.044715f*alu55*alu55*alu55))*-2.302208198144325f))))*alu55);
  data0_6946816[(alu36+1027)] = ((1/(1.0f+exp2(((alu56+(0.044715f*alu56*alu56*alu56))*-2.302208198144325f))))*alu56);
  var alu61 = (acc0[3]+val32);
  var alu62 = (acc0[7]+val33);
  var alu63 = (acc0[11]+val34);
  var alu64 = (acc0[15]+val35);
  data0_6946816[(alu36+1536)] = ((1/(1.0f+exp2(((alu61+(0.044715f*alu61*alu61*alu61))*-2.302208198144325f))))*alu61);
  data0_6946816[(alu36+1537)] = ((1/(1.0f+exp2(((alu62+(0.044715f*alu62*alu62*alu62))*-2.302208198144325f))))*alu62);
  data0_6946816[(alu36+1538)] = ((1/(1.0f+exp2(((alu63+(0.044715f*alu63*alu63*alu63))*-2.302208198144325f))))*alu63);
  data0_6946816[(alu36+1539)] = ((1/(1.0f+exp2(((alu64+(0.044715f*alu64*alu64*alu64))*-2.302208198144325f))))*alu64);
}`;

const r_53_2_16_4_4_128_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_27136:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_27136:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_108544:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 53 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    var cast3 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<11u))+cast3);
    var val0 = data2_108544[alu16];
    var alu17 = (bitcast<i32>((cast0<<15u))+bitcast<i32>((cast2<<11u))+cast3);
    var val1 = data3_65536[alu17];
    var val2 = data2_108544[(alu16+1)];
    var val3 = data3_65536[(alu17+1)];
    var val4 = data2_108544[(alu16+2)];
    var val5 = data3_65536[(alu17+2)];
    var val6 = data2_108544[(alu16+3)];
    var val7 = data3_65536[(alu17+3)];
    var val8 = data2_108544[(alu16+512)];
    var val9 = data2_108544[(alu16+513)];
    var val10 = data2_108544[(alu16+514)];
    var val11 = data2_108544[(alu16+515)];
    var val12 = data2_108544[(alu16+1024)];
    var val13 = data2_108544[(alu16+1025)];
    var val14 = data2_108544[(alu16+1026)];
    var val15 = data2_108544[(alu16+1027)];
    var val16 = data2_108544[(alu16+1536)];
    var val17 = data2_108544[(alu16+1537)];
    var val18 = data2_108544[(alu16+1538)];
    var val19 = data2_108544[(alu16+1539)];
    var val20 = data3_65536[(alu17+512)];
    var val21 = data3_65536[(alu17+513)];
    var val22 = data3_65536[(alu17+514)];
    var val23 = data3_65536[(alu17+515)];
    var val24 = data3_65536[(alu17+1024)];
    var val25 = data3_65536[(alu17+1025)];
    var val26 = data3_65536[(alu17+1026)];
    var val27 = data3_65536[(alu17+1027)];
    var val28 = data3_65536[(alu17+1536)];
    var val29 = data3_65536[(alu17+1537)];
    var val30 = data3_65536[(alu17+1538)];
    var val31 = data3_65536[(alu17+1539)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<9u)));
  var val32 = data1_27136[alu36];
  var val33 = data4_128[alu35];
  var alu37 = (alu36+1);
  var val34 = data1_27136[alu37];
  var val35 = data4_128[(alu35+1)];
  var alu38 = (alu36+2);
  var val36 = data1_27136[alu38];
  var val37 = data4_128[(alu35+2)];
  var alu39 = (alu36+3);
  var val38 = data1_27136[alu39];
  var val39 = data4_128[(alu35+3)];
  var alu40 = (alu36+128);
  var val40 = data1_27136[alu40];
  var alu41 = (alu36+129);
  var val41 = data1_27136[alu41];
  var alu42 = (alu36+130);
  var val42 = data1_27136[alu42];
  var alu43 = (alu36+131);
  var val43 = data1_27136[alu43];
  var alu44 = (alu36+256);
  var val44 = data1_27136[alu44];
  var alu45 = (alu36+257);
  var val45 = data1_27136[alu45];
  var alu46 = (alu36+258);
  var val46 = data1_27136[alu46];
  var alu47 = (alu36+259);
  var val47 = data1_27136[alu47];
  var alu48 = (alu36+384);
  var val48 = data1_27136[alu48];
  var alu49 = (alu36+385);
  var val49 = data1_27136[alu49];
  var alu50 = (alu36+386);
  var val50 = data1_27136[alu50];
  var alu51 = (alu36+387);
  var val51 = data1_27136[alu51];
  data0_27136[alu36] = (val32+acc0[0]+val33);
  data0_27136[alu37] = (val34+acc0[4]+val35);
  data0_27136[alu38] = (val36+acc0[8]+val37);
  data0_27136[alu39] = (val38+acc0[12]+val39);
  data0_27136[alu40] = (val40+acc0[1]+val33);
  data0_27136[alu41] = (val41+acc0[5]+val35);
  data0_27136[alu42] = (val42+acc0[9]+val37);
  data0_27136[alu43] = (val43+acc0[13]+val39);
  data0_27136[alu44] = (val44+acc0[2]+val33);
  data0_27136[alu45] = (val45+acc0[6]+val35);
  data0_27136[alu46] = (val46+acc0[10]+val37);
  data0_27136[alu47] = (val47+acc0[14]+val39);
  data0_27136[alu48] = (val48+acc0[3]+val33);
  data0_27136[alu49] = (val49+acc0[7]+val35);
  data0_27136[alu50] = (val50+acc0[11]+val37);
  data0_27136[alu51] = (val51+acc0[15]+val39);
}`;

const r_424_2_8_16_4_4_128_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1736704:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1736704:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_6946816:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 424 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
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
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    var cast4 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu16 = (bitcast<i32>((cast1<<14u))+bitcast<i32>((cast2<<11u))+cast4);
    var val0 = data2_6946816[alu16];
    var alu17 = (bitcast<i32>((cast0<<15u))+bitcast<i32>((cast3<<11u))+cast4);
    var val1 = data3_65536[alu17];
    var val2 = data2_6946816[(alu16+1)];
    var val3 = data3_65536[(alu17+1)];
    var val4 = data2_6946816[(alu16+2)];
    var val5 = data3_65536[(alu17+2)];
    var val6 = data2_6946816[(alu16+3)];
    var val7 = data3_65536[(alu17+3)];
    var val8 = data2_6946816[(alu16+512)];
    var val9 = data2_6946816[(alu16+513)];
    var val10 = data2_6946816[(alu16+514)];
    var val11 = data2_6946816[(alu16+515)];
    var val12 = data2_6946816[(alu16+1024)];
    var val13 = data2_6946816[(alu16+1025)];
    var val14 = data2_6946816[(alu16+1026)];
    var val15 = data2_6946816[(alu16+1027)];
    var val16 = data2_6946816[(alu16+1536)];
    var val17 = data2_6946816[(alu16+1537)];
    var val18 = data2_6946816[(alu16+1538)];
    var val19 = data2_6946816[(alu16+1539)];
    var val20 = data3_65536[(alu17+512)];
    var val21 = data3_65536[(alu17+513)];
    var val22 = data3_65536[(alu17+514)];
    var val23 = data3_65536[(alu17+515)];
    var val24 = data3_65536[(alu17+1024)];
    var val25 = data3_65536[(alu17+1025)];
    var val26 = data3_65536[(alu17+1026)];
    var val27 = data3_65536[(alu17+1027)];
    var val28 = data3_65536[(alu17+1536)];
    var val29 = data3_65536[(alu17+1537)];
    var val30 = data3_65536[(alu17+1538)];
    var val31 = data3_65536[(alu17+1539)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val1)+(val9*val3)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val1)+(val13*val3)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val16*val1)+(val17*val3)+(val18*val5)+(val19*val7));
    acc0[4] = (acc0[4]+(val0*val20)+(val2*val21)+(val4*val22)+(val6*val23));
    acc0[5] = (acc0[5]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[6] = (acc0[6]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[7] = (acc0[7]+(val16*val20)+(val17*val21)+(val18*val22)+(val19*val23));
    acc0[8] = (acc0[8]+(val0*val24)+(val2*val25)+(val4*val26)+(val6*val27));
    acc0[9] = (acc0[9]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[10] = (acc0[10]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    acc0[11] = (acc0[11]+(val16*val24)+(val17*val25)+(val18*val26)+(val19*val27));
    acc0[12] = (acc0[12]+(val0*val28)+(val2*val29)+(val4*val30)+(val6*val31));
    acc0[13] = (acc0[13]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    acc0[14] = (acc0[14]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    acc0[15] = (acc0[15]+(val16*val28)+(val17*val29)+(val18*val30)+(val19*val31));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<12u))+bitcast<i32>((cast2<<9u)));
  var val32 = data1_1736704[alu36];
  var val33 = data4_128[alu35];
  var alu37 = (alu36+1);
  var val34 = data1_1736704[alu37];
  var val35 = data4_128[(alu35+1)];
  var alu38 = (alu36+2);
  var val36 = data1_1736704[alu38];
  var val37 = data4_128[(alu35+2)];
  var alu39 = (alu36+3);
  var val38 = data1_1736704[alu39];
  var val39 = data4_128[(alu35+3)];
  var alu40 = (alu36+128);
  var val40 = data1_1736704[alu40];
  var alu41 = (alu36+129);
  var val41 = data1_1736704[alu41];
  var alu42 = (alu36+130);
  var val42 = data1_1736704[alu42];
  var alu43 = (alu36+131);
  var val43 = data1_1736704[alu43];
  var alu44 = (alu36+256);
  var val44 = data1_1736704[alu44];
  var alu45 = (alu36+257);
  var val45 = data1_1736704[alu45];
  var alu46 = (alu36+258);
  var val46 = data1_1736704[alu46];
  var alu47 = (alu36+259);
  var val47 = data1_1736704[alu47];
  var alu48 = (alu36+384);
  var val48 = data1_1736704[alu48];
  var alu49 = (alu36+385);
  var val49 = data1_1736704[alu49];
  var alu50 = (alu36+386);
  var val50 = data1_1736704[alu50];
  var alu51 = (alu36+387);
  var val51 = data1_1736704[alu51];
  data0_1736704[alu36] = (val32+acc0[0]+val33);
  data0_1736704[alu37] = (val34+acc0[4]+val35);
  data0_1736704[alu38] = (val36+acc0[8]+val37);
  data0_1736704[alu39] = (val38+acc0[12]+val39);
  data0_1736704[alu40] = (val40+acc0[1]+val33);
  data0_1736704[alu41] = (val41+acc0[5]+val35);
  data0_1736704[alu42] = (val42+acc0[9]+val37);
  data0_1736704[alu43] = (val43+acc0[13]+val39);
  data0_1736704[alu44] = (val44+acc0[2]+val33);
  data0_1736704[alu45] = (val45+acc0[6]+val35);
  data0_1736704[alu46] = (val46+acc0[10]+val37);
  data0_1736704[alu47] = (val47+acc0[14]+val39);
  data0_1736704[alu48] = (val48+acc0[3]+val33);
  data0_1736704[alu49] = (val49+acc0[7]+val35);
  data0_1736704[alu50] = (val50+acc0[11]+val37);
  data0_1736704[alu51] = (val51+acc0[15]+val39);
}`;

const E_16_4_8_16_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_128:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_27136:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1736704:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_13568:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_13568:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_128:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var alu0 = (lidx0+bitcast<i32>((cast0<<3u)));
  var val0 = data1_128[alu0];
  var val1 = data2_27136[alu0];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var val2 = data3_1736704[(alu0+(gidx0*434176)+(lidx1*27136)+12928)];
  var alu1 = ((gidx0*3392)+(lidx1*212)+101);
  var val3 = data4_13568[alu1];
  var val4 = data5_13568[alu1];
  var val5 = data6_128[alu0];
  var val6 = data7_128[alu0];
  var alu2 = (lidx1+bitcast<i32>((bitcast<u32>(gidx0)<<4u))+bitcast<i32>((cast0<<9u))+bitcast<i32>((bitcast<u32>(lidx0)<<6u)));
  data0_24576[alu2] = val0;
  data0_24576[(alu2+8192)] = val1;
  data0_24576[(alu2+16384)] = (((val2-val3)*val4*val5)+val6);
}`;

const E_16_16_8_3_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_384:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((gidx0*1536)+(lidx1*192)+bitcast<i32>((bitcast<u32>(lidx0)<<2u)));
  var val0 = data1_24576[(alu0+2)];
  var val1 = data1_24576[(alu0+3)];
  var val2 = data1_24576[(alu0+64)];
  var val3 = data1_24576[(alu0+65)];
  var val4 = data1_24576[(alu0+66)];
  var val5 = data1_24576[(alu0+67)];
  var val6 = data1_24576[(alu0+129)];
  var val7 = data1_24576[(alu0+130)];
  var val8 = data1_24576[(alu0+131)];
  var val9 = data1_24576[alu0];
  var alu1 = ((gidx0*24)+(lidx1*3));
  var val10 = data2_384[(alu1+1)];
  var val11 = data2_384[alu1];
  var val12 = data1_24576[(alu0+128)];
  var val13 = data2_384[(alu1+2)];
  var val14 = data1_24576[(alu0+1)];
  var alu2 = (alu1+(lidx0*1536));
  data0_24576[(alu2+384)] = (val14+val11);
  data0_24576[(alu2+385)] = (val3+val10);
  data0_24576[(alu2+386)] = (val6+val13);
  data0_24576[(alu2+768)] = (val0+val11);
  data0_24576[(alu2+769)] = (val4+val10);
  data0_24576[(alu2+770)] = (val7+val13);
  data0_24576[(alu2+1152)] = (val1+val11);
  data0_24576[(alu2+1153)] = (val5+val10);
  data0_24576[(alu2+1154)] = (val8+val13);
  data0_24576[(alu2+1)] = (val2+val10);
  data0_24576[(alu2+2)] = (val12+val13);
  data0_24576[alu2] = (val9+val11);
}`;

const r_8_8_8_16_3_3_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_73728:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_49152:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_384:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,9>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu9 = ((gidx1*3072)+(lidx0*384)+cast0);
    var val0 = data1_24576[(alu9+2)];
    var val1 = data1_24576[alu9];
    var alu10 = ((gidx0*6144)+(lidx1*384)+cast0);
    var val2 = data2_49152[(alu10+2)];
    var val3 = data2_49152[(alu10+129)];
    var val4 = data2_49152[(alu10+130)];
    var val5 = data2_49152[(alu10+131)];
    var val6 = data2_49152[alu10];
    var val7 = data1_24576[(alu9+1)];
    var val8 = data2_49152[(alu10+1)];
    var val9 = data1_24576[(alu9+3)];
    var val10 = data2_49152[(alu10+3)];
    var val11 = data1_24576[(alu9+128)];
    var val12 = data1_24576[(alu9+129)];
    var val13 = data1_24576[(alu9+130)];
    var val14 = data1_24576[(alu9+131)];
    var val15 = data1_24576[(alu9+256)];
    var val16 = data1_24576[(alu9+257)];
    var val17 = data1_24576[(alu9+258)];
    var val18 = data1_24576[(alu9+259)];
    var val19 = data2_49152[(alu10+128)];
    var val20 = data2_49152[(alu10+256)];
    var val21 = data2_49152[(alu10+257)];
    var val22 = data2_49152[(alu10+258)];
    var val23 = data2_49152[(alu10+259)];
    acc0[0] = (acc0[0]+(val1*val6)+(val7*val8)+(val0*val2)+(val9*val10));
    acc0[1] = (acc0[1]+(val11*val6)+(val12*val8)+(val13*val2)+(val14*val10));
    acc0[2] = (acc0[2]+(val15*val6)+(val16*val8)+(val17*val2)+(val18*val10));
    acc0[3] = (acc0[3]+(val1*val19)+(val7*val3)+(val0*val4)+(val9*val5));
    acc0[4] = (acc0[4]+(val11*val19)+(val12*val3)+(val13*val4)+(val14*val5));
    acc0[5] = (acc0[5]+(val15*val19)+(val16*val3)+(val17*val4)+(val18*val5));
    acc0[6] = (acc0[6]+(val1*val20)+(val7*val21)+(val0*val22)+(val9*val23));
    acc0[7] = (acc0[7]+(val11*val20)+(val12*val21)+(val13*val22)+(val14*val23));
    acc0[8] = (acc0[8]+(val15*val20)+(val16*val21)+(val17*val22)+(val18*val23));
  }
  var alu21 = ((gidx0*48)+(lidx1*3));
  var val24 = data3_384[(alu21+2)];
  var val25 = data3_384[alu21];
  var val26 = data3_384[(alu21+1)];
  var alu22 = (alu21+(gidx1*9216)+(lidx0*1152));
  data0_73728[(alu22+384)] = (acc0[1]+val25);
  data0_73728[(alu22+385)] = (acc0[4]+val26);
  data0_73728[(alu22+386)] = (acc0[7]+val24);
  data0_73728[(alu22+768)] = (acc0[2]+val25);
  data0_73728[(alu22+769)] = (acc0[5]+val26);
  data0_73728[(alu22+770)] = (acc0[8]+val24);
  data0_73728[(alu22+1)] = (acc0[3]+val26);
  data0_73728[(alu22+2)] = (acc0[6]+val24);
  data0_73728[alu22] = (acc0[0]+val25);
}`;

const r_4_16_8_3_3_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_4608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_73728:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((gidx0*18432)+(lidx0*1152)+bitcast<i32>((bitcast<u32>(lidx1)<<4u)));
  var val0 = data1_73728[(alu0+1)];
  var val1 = data1_73728[(alu0+2)];
  var val2 = data1_73728[(alu0+3)];
  var val3 = data1_73728[(alu0+4)];
  var val4 = data1_73728[(alu0+5)];
  var val5 = data1_73728[(alu0+6)];
  var val6 = data1_73728[(alu0+7)];
  var val7 = data1_73728[(alu0+8)];
  var val8 = data1_73728[(alu0+9)];
  var val9 = data1_73728[(alu0+10)];
  var val10 = data1_73728[(alu0+11)];
  var val11 = data1_73728[(alu0+12)];
  var val12 = data1_73728[(alu0+13)];
  var val13 = data1_73728[(alu0+14)];
  var val14 = data1_73728[(alu0+15)];
  var val15 = data1_73728[(alu0+128)];
  var val16 = data1_73728[(alu0+129)];
  var val17 = data1_73728[(alu0+130)];
  var val18 = data1_73728[(alu0+131)];
  var val19 = data1_73728[(alu0+132)];
  var val20 = data1_73728[(alu0+133)];
  var val21 = data1_73728[(alu0+134)];
  var val22 = data1_73728[(alu0+135)];
  var val23 = data1_73728[(alu0+136)];
  var val24 = data1_73728[(alu0+137)];
  var val25 = data1_73728[(alu0+138)];
  var val26 = data1_73728[(alu0+139)];
  var val27 = data1_73728[(alu0+140)];
  var val28 = data1_73728[(alu0+141)];
  var val29 = data1_73728[(alu0+142)];
  var val30 = data1_73728[(alu0+143)];
  var val31 = data1_73728[(alu0+384)];
  var val32 = data1_73728[(alu0+385)];
  var val33 = data1_73728[(alu0+386)];
  var val34 = data1_73728[(alu0+387)];
  var val35 = data1_73728[(alu0+388)];
  var val36 = data1_73728[(alu0+389)];
  var val37 = data1_73728[(alu0+390)];
  var val38 = data1_73728[(alu0+391)];
  var val39 = data1_73728[(alu0+392)];
  var val40 = data1_73728[(alu0+393)];
  var val41 = data1_73728[(alu0+394)];
  var val42 = data1_73728[(alu0+395)];
  var val43 = data1_73728[(alu0+396)];
  var val44 = data1_73728[(alu0+397)];
  var val45 = data1_73728[(alu0+398)];
  var val46 = data1_73728[(alu0+399)];
  var val47 = data1_73728[(alu0+512)];
  var val48 = data1_73728[(alu0+513)];
  var val49 = data1_73728[(alu0+514)];
  var val50 = data1_73728[(alu0+515)];
  var val51 = data1_73728[(alu0+516)];
  var val52 = data1_73728[(alu0+517)];
  var val53 = data1_73728[(alu0+518)];
  var val54 = data1_73728[(alu0+519)];
  var val55 = data1_73728[(alu0+520)];
  var val56 = data1_73728[(alu0+521)];
  var val57 = data1_73728[(alu0+522)];
  var val58 = data1_73728[(alu0+523)];
  var val59 = data1_73728[(alu0+524)];
  var val60 = data1_73728[(alu0+525)];
  var val61 = data1_73728[(alu0+526)];
  var val62 = data1_73728[(alu0+527)];
  var val63 = data1_73728[(alu0+768)];
  var val64 = data1_73728[(alu0+769)];
  var val65 = data1_73728[(alu0+770)];
  var val66 = data1_73728[(alu0+771)];
  var val67 = data1_73728[(alu0+772)];
  var val68 = data1_73728[(alu0+773)];
  var val69 = data1_73728[(alu0+774)];
  var val70 = data1_73728[(alu0+775)];
  var val71 = data1_73728[(alu0+776)];
  var val72 = data1_73728[(alu0+777)];
  var val73 = data1_73728[(alu0+778)];
  var val74 = data1_73728[(alu0+779)];
  var val75 = data1_73728[(alu0+780)];
  var val76 = data1_73728[(alu0+781)];
  var val77 = data1_73728[(alu0+782)];
  var val78 = data1_73728[(alu0+783)];
  var val79 = data1_73728[(alu0+896)];
  var val80 = data1_73728[(alu0+897)];
  var val81 = data1_73728[(alu0+898)];
  var val82 = data1_73728[(alu0+899)];
  var val83 = data1_73728[(alu0+900)];
  var val84 = data1_73728[(alu0+901)];
  var val85 = data1_73728[(alu0+902)];
  var val86 = data1_73728[(alu0+903)];
  var val87 = data1_73728[(alu0+904)];
  var val88 = data1_73728[(alu0+905)];
  var val89 = data1_73728[(alu0+906)];
  var val90 = data1_73728[(alu0+907)];
  var val91 = data1_73728[(alu0+908)];
  var val92 = data1_73728[(alu0+909)];
  var val93 = data1_73728[(alu0+910)];
  var val94 = data1_73728[(alu0+911)];
  var val95 = data1_73728[alu0];
  var alu1 = ((gidx0*1152)+(lidx0*72)+(lidx1*9));
  data0_4608[(alu1+1)] = (((val95*val47)+(val0*val48)+(val1*val49)+(val2*val50)+(val3*val51)+(val4*val52)+(val5*val53)+(val6*val54)+(val7*val55)+(val8*val56)+(val9*val57)+(val10*val58)+(val11*val59)+(val12*val60)+(val13*val61)+(val14*val62))*0.25f);
  data0_4608[(alu1+2)] = (((val95*val79)+(val0*val80)+(val1*val81)+(val2*val82)+(val3*val83)+(val4*val84)+(val5*val85)+(val6*val86)+(val7*val87)+(val8*val88)+(val9*val89)+(val10*val90)+(val11*val91)+(val12*val92)+(val13*val93)+(val14*val94))*0.25f);
  data0_4608[(alu1+3)] = (((val31*val15)+(val32*val16)+(val33*val17)+(val34*val18)+(val35*val19)+(val36*val20)+(val37*val21)+(val38*val22)+(val39*val23)+(val40*val24)+(val41*val25)+(val42*val26)+(val43*val27)+(val44*val28)+(val45*val29)+(val46*val30))*0.25f);
  data0_4608[(alu1+4)] = (((val31*val47)+(val32*val48)+(val33*val49)+(val34*val50)+(val35*val51)+(val36*val52)+(val37*val53)+(val38*val54)+(val39*val55)+(val40*val56)+(val41*val57)+(val42*val58)+(val43*val59)+(val44*val60)+(val45*val61)+(val46*val62))*0.25f);
  data0_4608[(alu1+5)] = (((val31*val79)+(val32*val80)+(val33*val81)+(val34*val82)+(val35*val83)+(val36*val84)+(val37*val85)+(val38*val86)+(val39*val87)+(val40*val88)+(val41*val89)+(val42*val90)+(val43*val91)+(val44*val92)+(val45*val93)+(val46*val94))*0.25f);
  data0_4608[(alu1+6)] = (((val63*val15)+(val64*val16)+(val65*val17)+(val66*val18)+(val67*val19)+(val68*val20)+(val69*val21)+(val70*val22)+(val71*val23)+(val72*val24)+(val73*val25)+(val74*val26)+(val75*val27)+(val76*val28)+(val77*val29)+(val78*val30))*0.25f);
  data0_4608[(alu1+7)] = (((val63*val47)+(val64*val48)+(val65*val49)+(val66*val50)+(val67*val51)+(val68*val52)+(val69*val53)+(val70*val54)+(val71*val55)+(val72*val56)+(val73*val57)+(val74*val58)+(val75*val59)+(val76*val60)+(val77*val61)+(val78*val62))*0.25f);
  data0_4608[(alu1+8)] = (((val63*val79)+(val64*val80)+(val65*val81)+(val66*val82)+(val67*val83)+(val68*val84)+(val69*val85)+(val70*val86)+(val71*val87)+(val72*val88)+(val73*val89)+(val74*val90)+(val75*val91)+(val76*val92)+(val77*val93)+(val78*val94))*0.25f);
  data0_4608[alu1] = (((val95*val15)+(val0*val16)+(val1*val17)+(val2*val18)+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22)+(val7*val23)+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27)+(val12*val28)+(val13*val29)+(val14*val30))*0.25f);
}`;

const r_16_32_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_4608:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0*288)+(lidx0*9));
  var val0 = data1_4608[(alu0+1)];
  var val1 = data1_4608[(alu0+2)];
  var val2 = data1_4608[(alu0+3)];
  var val3 = data1_4608[(alu0+4)];
  var val4 = data1_4608[(alu0+5)];
  var val5 = data1_4608[(alu0+6)];
  var val6 = data1_4608[(alu0+7)];
  var val7 = data1_4608[(alu0+8)];
  var val8 = data1_4608[alu0];
  var alu1 = ((gidx0*96)+(lidx0*3));
  var alu2 = select(val2,val3,(val2<val3));
  var alu3 = select(val5,val6,(val5<val6));
  var alu4 = select(val8,val0,(val8<val0));
  var alu5 = select(alu2,val4,(alu2<val4));
  var alu6 = select(alu3,val7,(alu3<val7));
  var alu7 = select(alu4,val1,(alu4<val1));
  data0_1536[(alu1+1)] = alu5;
  data0_1536[(alu1+2)] = alu6;
  data0_1536[alu1] = alu7;
}`;

const r_16_32_3_3n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_4608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 32 */
  var alu0 = ((gidx0*288)+(lidx0*9));
  var val0 = data1_4608[(alu0+1)];
  var val1 = data1_4608[(alu0+2)];
  var val2 = data1_4608[(alu0+3)];
  var val3 = data1_4608[alu0];
  var alu1 = ((gidx0*96)+(lidx0*3));
  var alu2 = (alu1+1);
  var val4 = data2_1536[alu2];
  var val5 = data2_1536[alu1];
  var val6 = data1_4608[(alu0+4)];
  var val7 = data1_4608[(alu0+5)];
  var val8 = data1_4608[(alu0+6)];
  var alu3 = (alu1+2);
  var val9 = data2_1536[alu3];
  var val10 = data1_4608[(alu0+7)];
  var val11 = data1_4608[(alu0+8)];
  data0_1536[alu2] = (exp2(((val2-val4)*1.4426950408889634f))+exp2(((val6-val4)*1.4426950408889634f))+exp2(((val7-val4)*1.4426950408889634f)));
  data0_1536[alu3] = (exp2(((val8-val9)*1.4426950408889634f))+exp2(((val10-val9)*1.4426950408889634f))+exp2(((val11-val9)*1.4426950408889634f)));
  data0_1536[alu1] = (exp2(((val3-val5)*1.4426950408889634f))+exp2(((val0-val5)*1.4426950408889634f))+exp2(((val1-val5)*1.4426950408889634f)));
}`;

const r_16_4_8_4_4_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_4608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_73728:array<f32>;
@compute @workgroup_size(4,8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var alu0 = ((gidx0*288)+(lidx0*72)+(lidx1*9));
  var val0 = data1_4608[alu0];
  var alu1 = ((gidx0*96)+(lidx0*24)+(lidx1*3));
  var val1 = data2_1536[alu1];
  var lidx2 = i32(lindex.z); /* 4 */
  var cast0 = bitcast<i32>((bitcast<u32>(lidx2)<<2u));
  var alu2 = ((gidx0*4608)+(lidx0*1152)+bitcast<i32>((bitcast<u32>(lidx1)<<4u))+cast0);
  var val2 = data4_73728[(alu2+256)];
  var val3 = data1_4608[(alu0+1)];
  var val4 = data4_73728[(alu2+257)];
  var val5 = data4_73728[(alu2+640)];
  var val6 = data1_4608[(alu0+2)];
  var val7 = data4_73728[(alu2+258)];
  var val8 = data4_73728[(alu2+259)];
  var val9 = data4_73728[(alu2+641)];
  var val10 = data4_73728[(alu2+642)];
  var val11 = data4_73728[(alu2+1024)];
  var val12 = data3_1536[alu1];
  var val13 = data4_73728[(alu2+643)];
  var val14 = data4_73728[(alu2+1025)];
  var val15 = data4_73728[(alu2+1026)];
  var val16 = data4_73728[(alu2+1027)];
  var val17 = data1_4608[(alu0+3)];
  var alu3 = (alu1+1);
  var val18 = data2_1536[alu3];
  var val19 = data1_4608[(alu0+4)];
  var val20 = data1_4608[(alu0+5)];
  var val21 = data3_1536[alu3];
  var val22 = data1_4608[(alu0+6)];
  var alu4 = (alu1+2);
  var val23 = data2_1536[alu4];
  var val24 = data1_4608[(alu0+7)];
  var val25 = data1_4608[(alu0+8)];
  var val26 = data3_1536[alu4];
  var alu5 = ((gidx0*1536)+(lidx0*384)+(lidx1*48)+cast0);
  var alu6 = exp2(((val17-val18)*1.4426950408889634f));
  var alu7 = exp2(((val19-val18)*1.4426950408889634f));
  var alu8 = exp2(((val20-val18)*1.4426950408889634f));
  var alu9 = (1/val21);
  data0_24576[(alu5+16)] = (((alu6*val2)+(alu7*val5)+(alu8*val11))*alu9);
  data0_24576[(alu5+17)] = (((alu6*val4)+(alu7*val9)+(alu8*val14))*alu9);
  data0_24576[(alu5+18)] = (((alu6*val7)+(alu7*val10)+(alu8*val15))*alu9);
  data0_24576[(alu5+19)] = (((alu6*val8)+(alu7*val13)+(alu8*val16))*alu9);
  var alu14 = exp2(((val22-val23)*1.4426950408889634f));
  var alu15 = exp2(((val24-val23)*1.4426950408889634f));
  var alu16 = exp2(((val25-val23)*1.4426950408889634f));
  var alu17 = (1/val26);
  data0_24576[(alu5+32)] = (((alu14*val2)+(alu15*val5)+(alu16*val11))*alu17);
  data0_24576[(alu5+33)] = (((alu14*val4)+(alu15*val9)+(alu16*val14))*alu17);
  data0_24576[(alu5+34)] = (((alu14*val7)+(alu15*val10)+(alu16*val15))*alu17);
  data0_24576[(alu5+35)] = (((alu14*val8)+(alu15*val13)+(alu16*val16))*alu17);
  var alu22 = exp2(((val3-val1)*1.4426950408889634f));
  var alu23 = exp2(((val6-val1)*1.4426950408889634f));
  var alu24 = exp2(((val0-val1)*1.4426950408889634f));
  var alu25 = (1/val12);
  data0_24576[(alu5+1)] = (((alu24*val4)+(alu22*val9)+(alu23*val14))*alu25);
  data0_24576[(alu5+2)] = (((alu24*val7)+(alu22*val10)+(alu23*val15))*alu25);
  data0_24576[(alu5+3)] = (((alu24*val8)+(alu22*val13)+(alu23*val16))*alu25);
  data0_24576[alu5] = (((alu24*val2)+(alu22*val5)+(alu23*val11))*alu25);
}`;

const r_8_2_8_16_4_3_8_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_24576:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_16384:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  var alu0 = ((gidx1*3072)+(lidx0*384));
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
  for (var Ridx0_0 = 0; Ridx0_0 < 8; Ridx0_0++) {
    var alu13 = (alu0+(Ridx0_0*48));
    var val0 = data2_24576[alu13];
    var alu14 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<9u))+bitcast<i32>((bitcast<u32>(Ridx0_0)<<4u)));
    var val1 = data3_16384[alu14];
    var val2 = data2_24576[(alu13+1)];
    var val3 = data3_16384[(alu14+1)];
    var val4 = data2_24576[(alu13+2)];
    var val5 = data3_16384[(alu14+2)];
    var val6 = data2_24576[(alu13+3)];
    var val7 = data3_16384[(alu14+3)];
    var val8 = data2_24576[(alu13+4)];
    var val9 = data3_16384[(alu14+4)];
    var val10 = data2_24576[(alu13+5)];
    var val11 = data3_16384[(alu14+5)];
    var val12 = data2_24576[(alu13+6)];
    var val13 = data3_16384[(alu14+6)];
    var val14 = data2_24576[(alu13+7)];
    var val15 = data3_16384[(alu14+7)];
    var val16 = data2_24576[(alu13+8)];
    var val17 = data3_16384[(alu14+8)];
    var val18 = data2_24576[(alu13+9)];
    var val19 = data3_16384[(alu14+9)];
    var val20 = data2_24576[(alu13+10)];
    var val21 = data3_16384[(alu14+10)];
    var val22 = data2_24576[(alu13+11)];
    var val23 = data3_16384[(alu14+11)];
    var val24 = data2_24576[(alu13+12)];
    var val25 = data3_16384[(alu14+12)];
    var val26 = data2_24576[(alu13+13)];
    var val27 = data3_16384[(alu14+13)];
    var val28 = data2_24576[(alu13+14)];
    var val29 = data3_16384[(alu14+14)];
    var val30 = data2_24576[(alu13+15)];
    var val31 = data3_16384[(alu14+15)];
    var val32 = data2_24576[(alu13+16)];
    var val33 = data2_24576[(alu13+17)];
    var val34 = data2_24576[(alu13+18)];
    var val35 = data2_24576[(alu13+19)];
    var val36 = data2_24576[(alu13+20)];
    var val37 = data2_24576[(alu13+21)];
    var val38 = data2_24576[(alu13+22)];
    var val39 = data2_24576[(alu13+23)];
    var val40 = data2_24576[(alu13+24)];
    var val41 = data2_24576[(alu13+25)];
    var val42 = data2_24576[(alu13+26)];
    var val43 = data2_24576[(alu13+27)];
    var val44 = data2_24576[(alu13+28)];
    var val45 = data2_24576[(alu13+29)];
    var val46 = data2_24576[(alu13+30)];
    var val47 = data2_24576[(alu13+31)];
    var val48 = data2_24576[(alu13+32)];
    var val49 = data2_24576[(alu13+33)];
    var val50 = data2_24576[(alu13+34)];
    var val51 = data2_24576[(alu13+35)];
    var val52 = data2_24576[(alu13+36)];
    var val53 = data2_24576[(alu13+37)];
    var val54 = data2_24576[(alu13+38)];
    var val55 = data2_24576[(alu13+39)];
    var val56 = data2_24576[(alu13+40)];
    var val57 = data2_24576[(alu13+41)];
    var val58 = data2_24576[(alu13+42)];
    var val59 = data2_24576[(alu13+43)];
    var val60 = data2_24576[(alu13+44)];
    var val61 = data2_24576[(alu13+45)];
    var val62 = data2_24576[(alu13+46)];
    var val63 = data2_24576[(alu13+47)];
    var val64 = data3_16384[(alu14+128)];
    var val65 = data3_16384[(alu14+129)];
    var val66 = data3_16384[(alu14+130)];
    var val67 = data3_16384[(alu14+131)];
    var val68 = data3_16384[(alu14+132)];
    var val69 = data3_16384[(alu14+133)];
    var val70 = data3_16384[(alu14+134)];
    var val71 = data3_16384[(alu14+135)];
    var val72 = data3_16384[(alu14+136)];
    var val73 = data3_16384[(alu14+137)];
    var val74 = data3_16384[(alu14+138)];
    var val75 = data3_16384[(alu14+139)];
    var val76 = data3_16384[(alu14+140)];
    var val77 = data3_16384[(alu14+141)];
    var val78 = data3_16384[(alu14+142)];
    var val79 = data3_16384[(alu14+143)];
    var val80 = data3_16384[(alu14+256)];
    var val81 = data3_16384[(alu14+257)];
    var val82 = data3_16384[(alu14+258)];
    var val83 = data3_16384[(alu14+259)];
    var val84 = data3_16384[(alu14+260)];
    var val85 = data3_16384[(alu14+261)];
    var val86 = data3_16384[(alu14+262)];
    var val87 = data3_16384[(alu14+263)];
    var val88 = data3_16384[(alu14+264)];
    var val89 = data3_16384[(alu14+265)];
    var val90 = data3_16384[(alu14+266)];
    var val91 = data3_16384[(alu14+267)];
    var val92 = data3_16384[(alu14+268)];
    var val93 = data3_16384[(alu14+269)];
    var val94 = data3_16384[(alu14+270)];
    var val95 = data3_16384[(alu14+271)];
    var val96 = data3_16384[(alu14+384)];
    var val97 = data3_16384[(alu14+385)];
    var val98 = data3_16384[(alu14+386)];
    var val99 = data3_16384[(alu14+387)];
    var val100 = data3_16384[(alu14+388)];
    var val101 = data3_16384[(alu14+389)];
    var val102 = data3_16384[(alu14+390)];
    var val103 = data3_16384[(alu14+391)];
    var val104 = data3_16384[(alu14+392)];
    var val105 = data3_16384[(alu14+393)];
    var val106 = data3_16384[(alu14+394)];
    var val107 = data3_16384[(alu14+395)];
    var val108 = data3_16384[(alu14+396)];
    var val109 = data3_16384[(alu14+397)];
    var val110 = data3_16384[(alu14+398)];
    var val111 = data3_16384[(alu14+399)];
    acc0[0] = (acc0[0]+(val0*val1)+(val2*val3)+(val4*val5)+(val6*val7)+(val8*val9)+(val10*val11)+(val12*val13)+(val14*val15)+(val16*val17)+(val18*val19)+(val20*val21)+(val22*val23)+(val24*val25)+(val26*val27)+(val28*val29)+(val30*val31));
    acc0[1] = (acc0[1]+(val32*val1)+(val33*val3)+(val34*val5)+(val35*val7)+(val36*val9)+(val37*val11)+(val38*val13)+(val39*val15)+(val40*val17)+(val41*val19)+(val42*val21)+(val43*val23)+(val44*val25)+(val45*val27)+(val46*val29)+(val47*val31));
    acc0[2] = (acc0[2]+(val48*val1)+(val49*val3)+(val50*val5)+(val51*val7)+(val52*val9)+(val53*val11)+(val54*val13)+(val55*val15)+(val56*val17)+(val57*val19)+(val58*val21)+(val59*val23)+(val60*val25)+(val61*val27)+(val62*val29)+(val63*val31));
    acc0[3] = (acc0[3]+(val0*val64)+(val2*val65)+(val4*val66)+(val6*val67)+(val8*val68)+(val10*val69)+(val12*val70)+(val14*val71)+(val16*val72)+(val18*val73)+(val20*val74)+(val22*val75)+(val24*val76)+(val26*val77)+(val28*val78)+(val30*val79));
    acc0[4] = (acc0[4]+(val32*val64)+(val33*val65)+(val34*val66)+(val35*val67)+(val36*val68)+(val37*val69)+(val38*val70)+(val39*val71)+(val40*val72)+(val41*val73)+(val42*val74)+(val43*val75)+(val44*val76)+(val45*val77)+(val46*val78)+(val47*val79));
    acc0[5] = (acc0[5]+(val48*val64)+(val49*val65)+(val50*val66)+(val51*val67)+(val52*val68)+(val53*val69)+(val54*val70)+(val55*val71)+(val56*val72)+(val57*val73)+(val58*val74)+(val59*val75)+(val60*val76)+(val61*val77)+(val62*val78)+(val63*val79));
    acc0[6] = (acc0[6]+(val0*val80)+(val2*val81)+(val4*val82)+(val6*val83)+(val8*val84)+(val10*val85)+(val12*val86)+(val14*val87)+(val16*val88)+(val18*val89)+(val20*val90)+(val22*val91)+(val24*val92)+(val26*val93)+(val28*val94)+(val30*val95));
    acc0[7] = (acc0[7]+(val32*val80)+(val33*val81)+(val34*val82)+(val35*val83)+(val36*val84)+(val37*val85)+(val38*val86)+(val39*val87)+(val40*val88)+(val41*val89)+(val42*val90)+(val43*val91)+(val44*val92)+(val45*val93)+(val46*val94)+(val47*val95));
    acc0[8] = (acc0[8]+(val48*val80)+(val49*val81)+(val50*val82)+(val51*val83)+(val52*val84)+(val53*val85)+(val54*val86)+(val55*val87)+(val56*val88)+(val57*val89)+(val58*val90)+(val59*val91)+(val60*val92)+(val61*val93)+(val62*val94)+(val63*val95));
    acc0[9] = (acc0[9]+(val0*val96)+(val2*val97)+(val4*val98)+(val6*val99)+(val8*val100)+(val10*val101)+(val12*val102)+(val14*val103)+(val16*val104)+(val18*val105)+(val20*val106)+(val22*val107)+(val24*val108)+(val26*val109)+(val28*val110)+(val30*val111));
    acc0[10] = (acc0[10]+(val32*val96)+(val33*val97)+(val34*val98)+(val35*val99)+(val36*val100)+(val37*val101)+(val38*val102)+(val39*val103)+(val40*val104)+(val41*val105)+(val42*val106)+(val43*val107)+(val44*val108)+(val45*val109)+(val46*val110)+(val47*val111));
    acc0[11] = (acc0[11]+(val48*val96)+(val49*val97)+(val50*val98)+(val51*val99)+(val52*val100)+(val53*val101)+(val54*val102)+(val55*val103)+(val56*val104)+(val57*val105)+(val58*val106)+(val59*val107)+(val60*val108)+(val61*val109)+(val62*val110)+(val63*val111));
  }
  var alu28 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var alu29 = (alu28+alu0);
  var val112 = data1_24576[alu29];
  var val113 = data4_128[alu28];
  var alu30 = (alu29+1);
  var val114 = data1_24576[alu30];
  var val115 = data4_128[(alu28+1)];
  var alu31 = (alu29+2);
  var val116 = data1_24576[alu31];
  var val117 = data4_128[(alu28+2)];
  var alu32 = (alu29+3);
  var val118 = data1_24576[alu32];
  var val119 = data4_128[(alu28+3)];
  var alu33 = (alu29+128);
  var val120 = data1_24576[alu33];
  var alu34 = (alu29+129);
  var val121 = data1_24576[alu34];
  var alu35 = (alu29+130);
  var val122 = data1_24576[alu35];
  var alu36 = (alu29+131);
  var val123 = data1_24576[alu36];
  var alu37 = (alu29+256);
  var val124 = data1_24576[alu37];
  var alu38 = (alu29+257);
  var val125 = data1_24576[alu38];
  var alu39 = (alu29+258);
  var val126 = data1_24576[alu39];
  var alu40 = (alu29+259);
  var val127 = data1_24576[alu40];
  data0_24576[alu29] = (val112+acc0[0]+val113);
  data0_24576[alu30] = (val114+acc0[3]+val115);
  data0_24576[alu31] = (val116+acc0[6]+val117);
  data0_24576[alu32] = (val118+acc0[9]+val119);
  data0_24576[alu33] = (val120+acc0[1]+val113);
  data0_24576[alu34] = (val121+acc0[4]+val115);
  data0_24576[alu35] = (val122+acc0[7]+val117);
  data0_24576[alu36] = (val123+acc0[10]+val119);
  data0_24576[alu37] = (val124+acc0[2]+val113);
  data0_24576[alu38] = (val125+acc0[5]+val115);
  data0_24576[alu39] = (val126+acc0[8]+val117);
  data0_24576[alu40] = (val127+acc0[11]+val119);
}`;

const r_192_16_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_192:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 192 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val0 = data1_24576[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
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
    data0_192[gidx0] = (acc1[0]*0.0078125f);
  }
}`;

const r_192_16_8n1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_192:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_192:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 192 */
  var val0 = data2_192[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var val1 = data1_24576[(bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0+bitcast<i32>((bitcast<u32>(gidx0)<<7u)))];
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
    data0_192[gidx0] = (1/sqrt(((acc1[0]*0.0078125f)+1e-05f)));
  }
}`;

const E_8_2_8_16_4_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_192:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_192:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+(gidx1*3072)+(lidx0*384));
  var val0 = data1_24576[alu1];
  var alu2 = ((gidx1*24)+(lidx0*3));
  var val1 = data2_192[alu2];
  var val2 = data3_192[alu2];
  var val3 = data4_128[alu0];
  var val4 = data5_128[alu0];
  var alu3 = (alu1+1);
  var val5 = data1_24576[alu3];
  var alu4 = (alu0+1);
  var val6 = data4_128[alu4];
  var val7 = data5_128[alu4];
  var alu5 = (alu1+2);
  var val8 = data1_24576[alu5];
  var alu6 = (alu0+2);
  var val9 = data4_128[alu6];
  var val10 = data5_128[alu6];
  var alu7 = (alu1+3);
  var val11 = data1_24576[alu7];
  var alu8 = (alu0+3);
  var val12 = data4_128[alu8];
  var val13 = data5_128[alu8];
  var alu9 = (alu1+128);
  var val14 = data1_24576[alu9];
  var alu10 = (alu2+1);
  var val15 = data2_192[alu10];
  var val16 = data3_192[alu10];
  var alu11 = (alu1+129);
  var val17 = data1_24576[alu11];
  var alu12 = (alu1+130);
  var val18 = data1_24576[alu12];
  var alu13 = (alu1+131);
  var val19 = data1_24576[alu13];
  var alu14 = (alu1+256);
  var val20 = data1_24576[alu14];
  var alu15 = (alu2+2);
  var val21 = data2_192[alu15];
  var val22 = data3_192[alu15];
  var alu16 = (alu1+257);
  var val23 = data1_24576[alu16];
  var alu17 = (alu1+258);
  var val24 = data1_24576[alu17];
  var alu18 = (alu1+259);
  var val25 = data1_24576[alu18];
  data0_24576[alu1] = (((val0-val1)*val2*val3)+val4);
  data0_24576[alu3] = (((val5-val1)*val2*val6)+val7);
  data0_24576[alu5] = (((val8-val1)*val2*val9)+val10);
  data0_24576[alu7] = (((val11-val1)*val2*val12)+val13);
  data0_24576[alu9] = (((val14-val15)*val16*val3)+val4);
  data0_24576[alu11] = (((val17-val15)*val16*val6)+val7);
  data0_24576[alu12] = (((val18-val15)*val16*val9)+val10);
  data0_24576[alu13] = (((val19-val15)*val16*val12)+val13);
  data0_24576[alu14] = (((val20-val21)*val22*val3)+val4);
  data0_24576[alu16] = (((val23-val21)*val22*val6)+val7);
  data0_24576[alu17] = (((val24-val21)*val22*val9)+val10);
  data0_24576[alu18] = (((val25-val21)*val22*val12)+val13);
}`;

const r_8_8_8_16_4_3_32_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_98304:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 8 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
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
    var cast2 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu12 = ((gidx1*3072)+(lidx0*384)+cast2);
    var val0 = data1_24576[(alu12+2)];
    var val1 = data1_24576[alu12];
    var alu13 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<9u))+cast2);
    var val2 = data2_65536[alu13];
    var val3 = data1_24576[(alu12+1)];
    var val4 = data2_65536[(alu13+1)];
    var val5 = data2_65536[(alu13+2)];
    var val6 = data1_24576[(alu12+3)];
    var val7 = data2_65536[(alu13+3)];
    var val8 = data1_24576[(alu12+128)];
    var val9 = data1_24576[(alu12+129)];
    var val10 = data1_24576[(alu12+130)];
    var val11 = data1_24576[(alu12+131)];
    var val12 = data1_24576[(alu12+256)];
    var val13 = data1_24576[(alu12+257)];
    var val14 = data1_24576[(alu12+258)];
    var val15 = data1_24576[(alu12+259)];
    var val16 = data2_65536[(alu13+128)];
    var val17 = data2_65536[(alu13+129)];
    var val18 = data2_65536[(alu13+130)];
    var val19 = data2_65536[(alu13+131)];
    var val20 = data2_65536[(alu13+256)];
    var val21 = data2_65536[(alu13+257)];
    var val22 = data2_65536[(alu13+258)];
    var val23 = data2_65536[(alu13+259)];
    var val24 = data2_65536[(alu13+384)];
    var val25 = data2_65536[(alu13+385)];
    var val26 = data2_65536[(alu13+386)];
    var val27 = data2_65536[(alu13+387)];
    acc0[0] = (acc0[0]+(val1*val2)+(val3*val4)+(val0*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val2)+(val9*val4)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val2)+(val13*val4)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val1*val16)+(val3*val17)+(val0*val18)+(val6*val19));
    acc0[4] = (acc0[4]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val19));
    acc0[5] = (acc0[5]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val19));
    acc0[6] = (acc0[6]+(val1*val20)+(val3*val21)+(val0*val22)+(val6*val23));
    acc0[7] = (acc0[7]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[8] = (acc0[8]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[9] = (acc0[9]+(val1*val24)+(val3*val25)+(val0*val26)+(val6*val27));
    acc0[10] = (acc0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[11] = (acc0[11]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
  }
  var alu27 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val28 = data3_512[alu27];
  var val29 = data3_512[(alu27+1)];
  var val30 = data3_512[(alu27+2)];
  var val31 = data3_512[(alu27+3)];
  var alu28 = (alu27+(gidx1*12288)+(lidx0*1536));
  var alu29 = (acc0[0]+val28);
  var alu30 = (acc0[3]+val29);
  var alu31 = (acc0[6]+val30);
  var alu32 = (acc0[9]+val31);
  data0_98304[alu28] = ((1/(1.0f+exp2(((alu29+(0.044715f*alu29*alu29*alu29))*-2.302208198144325f))))*alu29);
  data0_98304[(alu28+1)] = ((1/(1.0f+exp2(((alu30+(0.044715f*alu30*alu30*alu30))*-2.302208198144325f))))*alu30);
  data0_98304[(alu28+2)] = ((1/(1.0f+exp2(((alu31+(0.044715f*alu31*alu31*alu31))*-2.302208198144325f))))*alu31);
  data0_98304[(alu28+3)] = ((1/(1.0f+exp2(((alu32+(0.044715f*alu32*alu32*alu32))*-2.302208198144325f))))*alu32);
  var alu37 = (acc0[1]+val28);
  var alu38 = (acc0[4]+val29);
  var alu39 = (acc0[7]+val30);
  var alu40 = (acc0[10]+val31);
  data0_98304[(alu28+512)] = ((1/(1.0f+exp2(((alu37+(0.044715f*alu37*alu37*alu37))*-2.302208198144325f))))*alu37);
  data0_98304[(alu28+513)] = ((1/(1.0f+exp2(((alu38+(0.044715f*alu38*alu38*alu38))*-2.302208198144325f))))*alu38);
  data0_98304[(alu28+514)] = ((1/(1.0f+exp2(((alu39+(0.044715f*alu39*alu39*alu39))*-2.302208198144325f))))*alu39);
  data0_98304[(alu28+515)] = ((1/(1.0f+exp2(((alu40+(0.044715f*alu40*alu40*alu40))*-2.302208198144325f))))*alu40);
  var alu45 = (acc0[2]+val28);
  var alu46 = (acc0[5]+val29);
  var alu47 = (acc0[8]+val30);
  var alu48 = (acc0[11]+val31);
  data0_98304[(alu28+1024)] = ((1/(1.0f+exp2(((alu45+(0.044715f*alu45*alu45*alu45))*-2.302208198144325f))))*alu45);
  data0_98304[(alu28+1025)] = ((1/(1.0f+exp2(((alu46+(0.044715f*alu46*alu46*alu46))*-2.302208198144325f))))*alu46);
  data0_98304[(alu28+1026)] = ((1/(1.0f+exp2(((alu47+(0.044715f*alu47*alu47*alu47))*-2.302208198144325f))))*alu47);
  data0_98304[(alu28+1027)] = ((1/(1.0f+exp2(((alu48+(0.044715f*alu48*alu48*alu48))*-2.302208198144325f))))*alu48);
}`;

const r_8_2_8_16_4_3_128_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_24576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_98304:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
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
  for (var Ridx0 = 0; Ridx0 < 128; Ridx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Ridx0)<<2u));
    var alu12 = ((gidx1*12288)+(lidx0*1536)+cast2);
    var val0 = data2_98304[(alu12+2)];
    var val1 = data2_98304[alu12];
    var alu13 = (bitcast<i32>((cast0<<15u))+bitcast<i32>((cast1<<11u))+cast2);
    var val2 = data3_65536[alu13];
    var val3 = data2_98304[(alu12+1)];
    var val4 = data3_65536[(alu13+1)];
    var val5 = data3_65536[(alu13+2)];
    var val6 = data2_98304[(alu12+3)];
    var val7 = data3_65536[(alu13+3)];
    var val8 = data2_98304[(alu12+512)];
    var val9 = data2_98304[(alu12+513)];
    var val10 = data2_98304[(alu12+514)];
    var val11 = data2_98304[(alu12+515)];
    var val12 = data2_98304[(alu12+1024)];
    var val13 = data2_98304[(alu12+1025)];
    var val14 = data2_98304[(alu12+1026)];
    var val15 = data2_98304[(alu12+1027)];
    var val16 = data3_65536[(alu13+512)];
    var val17 = data3_65536[(alu13+513)];
    var val18 = data3_65536[(alu13+514)];
    var val19 = data3_65536[(alu13+515)];
    var val20 = data3_65536[(alu13+1024)];
    var val21 = data3_65536[(alu13+1025)];
    var val22 = data3_65536[(alu13+1026)];
    var val23 = data3_65536[(alu13+1027)];
    var val24 = data3_65536[(alu13+1536)];
    var val25 = data3_65536[(alu13+1537)];
    var val26 = data3_65536[(alu13+1538)];
    var val27 = data3_65536[(alu13+1539)];
    acc0[0] = (acc0[0]+(val1*val2)+(val3*val4)+(val0*val5)+(val6*val7));
    acc0[1] = (acc0[1]+(val8*val2)+(val9*val4)+(val10*val5)+(val11*val7));
    acc0[2] = (acc0[2]+(val12*val2)+(val13*val4)+(val14*val5)+(val15*val7));
    acc0[3] = (acc0[3]+(val1*val16)+(val3*val17)+(val0*val18)+(val6*val19));
    acc0[4] = (acc0[4]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val19));
    acc0[5] = (acc0[5]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val19));
    acc0[6] = (acc0[6]+(val1*val20)+(val3*val21)+(val0*val22)+(val6*val23));
    acc0[7] = (acc0[7]+(val8*val20)+(val9*val21)+(val10*val22)+(val11*val23));
    acc0[8] = (acc0[8]+(val12*val20)+(val13*val21)+(val14*val22)+(val15*val23));
    acc0[9] = (acc0[9]+(val1*val24)+(val3*val25)+(val0*val26)+(val6*val27));
    acc0[10] = (acc0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    acc0[11] = (acc0[11]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
  }
  var alu27 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var alu28 = (alu27+(gidx1*3072)+(lidx0*384));
  var val28 = data1_24576[alu28];
  var val29 = data4_128[alu27];
  var alu29 = (alu28+1);
  var val30 = data1_24576[alu29];
  var val31 = data4_128[(alu27+1)];
  var alu30 = (alu28+2);
  var val32 = data1_24576[alu30];
  var val33 = data4_128[(alu27+2)];
  var alu31 = (alu28+3);
  var val34 = data1_24576[alu31];
  var val35 = data4_128[(alu27+3)];
  var alu32 = (alu28+128);
  var val36 = data1_24576[alu32];
  var alu33 = (alu28+129);
  var val37 = data1_24576[alu33];
  var alu34 = (alu28+130);
  var val38 = data1_24576[alu34];
  var alu35 = (alu28+131);
  var val39 = data1_24576[alu35];
  var alu36 = (alu28+256);
  var val40 = data1_24576[alu36];
  var alu37 = (alu28+257);
  var val41 = data1_24576[alu37];
  var alu38 = (alu28+258);
  var val42 = data1_24576[alu38];
  var alu39 = (alu28+259);
  var val43 = data1_24576[alu39];
  data0_24576[alu28] = (val28+acc0[0]+val29);
  data0_24576[alu29] = (val30+acc0[3]+val31);
  data0_24576[alu30] = (val32+acc0[6]+val33);
  data0_24576[alu31] = (val34+acc0[9]+val35);
  data0_24576[alu32] = (val36+acc0[1]+val29);
  data0_24576[alu33] = (val37+acc0[4]+val31);
  data0_24576[alu34] = (val38+acc0[7]+val33);
  data0_24576[alu35] = (val39+acc0[10]+val35);
  data0_24576[alu36] = (val40+acc0[2]+val29);
  data0_24576[alu37] = (val41+acc0[5]+val31);
  data0_24576[alu38] = (val42+acc0[8]+val33);
  data0_24576[alu39] = (val43+acc0[11]+val35);
}`;

const r_64_16_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_64:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_24576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_192:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_192:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_128:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_128:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_128:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7_1:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 64 */
  var alu0 = (gidx0*3);
  var val0 = data2_192[alu0];
  var val1 = data3_192[alu0];
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  for (var Ridx0 = 0; Ridx0 < 8; Ridx0++) {
    var alu2 = (bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Ridx0);
    var val2 = data1_24576[(alu2+(gidx0*384))];
    var val3 = data4_128[alu2];
    var val4 = data5_128[alu2];
    var val5 = data6_128[alu2];
    acc0[0] = (acc0[0]+((((val2-val0)*val1*val3)+val4)*val5));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  acc1[0] = 0.0f;
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val6 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val6);
  }
  var val7 = data7_1[0];
  var alu10 = ((bool(lidx0))!=true);
  if (alu10) {
    data0_64[gidx0] = ((acc1[0]+val7)*10.0f);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 108544);;
    const input0 = createEmptyBuf(device, 848);;
    const buf_1 = createWeightBuf(device, 48128, getTensorBuffer(safetensor, metadata['enc.tok_emb.weight']));
    const buf_2 = createWeightBuf(device, 108544, getTensorBuffer(safetensor, metadata['enc.pos_emb.weight']));
    const input1 = createEmptyBuf(device, 848);;
    const buf_3 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['enc.la_emb.weight']));
    const buf_4 = createEmptyBuf(device, 6946816);;
    const input2 = createEmptyBuf(device, 54272);;
    const input3 = createEmptyBuf(device, 54272);;
    const buf_5 = createEmptyBuf(device, 325632);;
    const buf_6 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.in_proj_weight']));
    const buf_7 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.in_proj_bias']));
    const buf_8 = createEmptyBuf(device, 20840448);;
    const buf_9 = createEmptyBuf(device, 1438208);;
    const buf_10 = createEmptyBuf(device, 92045312);;
    const buf_11 = createEmptyBuf(device, 6784);;
    const buf_12 = createEmptyBuf(device, 434176);;
    const buf_13 = createEmptyBuf(device, 6784);;
    const buf_14 = createEmptyBuf(device, 434176);;
    const buf_15 = createEmptyBuf(device, 108544);;
    const buf_16 = createEmptyBuf(device, 6946816);;
    const buf_17 = createEmptyBuf(device, 108544);;
    const buf_18 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.out_proj.weight']));
    const buf_19 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.self_attn.out_proj.bias']));
    const buf_20 = createEmptyBuf(device, 6946816);;
    const buf_21 = createEmptyBuf(device, 848);;
    const buf_22 = createEmptyBuf(device, 54272);;
    const buf_23 = createEmptyBuf(device, 848);;
    const buf_24 = createEmptyBuf(device, 54272);;
    const buf_25 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.norm1.weight']));
    const buf_26 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.norm1.bias']));
    const buf_27 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.0.linear1.weight']));
    const buf_28 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['enc.layers.0.linear1.bias']));
    const buf_29 = createEmptyBuf(device, 27787264);;
    const buf_30 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.0.linear2.weight']));
    const buf_31 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.linear2.bias']));
    const buf_32 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.norm2.weight']));
    const buf_33 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.0.norm2.bias']));
    const buf_34 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.in_proj_weight']));
    const buf_35 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.in_proj_bias']));
    const buf_36 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.out_proj.weight']));
    const buf_37 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.self_attn.out_proj.bias']));
    const buf_38 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.norm1.weight']));
    const buf_39 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.norm1.bias']));
    const buf_40 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.1.linear1.weight']));
    const buf_41 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['enc.layers.1.linear1.bias']));
    const buf_42 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.1.linear2.weight']));
    const buf_43 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.linear2.bias']));
    const buf_44 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.norm2.weight']));
    const buf_45 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.1.norm2.bias']));
    const buf_46 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.in_proj_weight']));
    const buf_47 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.in_proj_bias']));
    const buf_48 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.out_proj.weight']));
    const buf_49 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.self_attn.out_proj.bias']));
    const buf_50 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.norm1.weight']));
    const buf_51 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.norm1.bias']));
    const buf_52 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.2.linear1.weight']));
    const buf_53 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['enc.layers.2.linear1.bias']));
    const buf_54 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.2.linear2.weight']));
    const buf_55 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.linear2.bias']));
    const buf_56 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.norm2.weight']));
    const buf_57 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.2.norm2.bias']));
    const buf_58 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.in_proj_weight']));
    const buf_59 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.in_proj_bias']));
    const buf_60 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.out_proj.weight']));
    const buf_61 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.self_attn.out_proj.bias']));
    const buf_62 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.norm1.weight']));
    const buf_63 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.norm1.bias']));
    const buf_64 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.3.linear1.weight']));
    const buf_65 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['enc.layers.3.linear1.bias']));
    const buf_66 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['enc.layers.3.linear2.weight']));
    const buf_67 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.linear2.bias']));
    const buf_68 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.norm2.weight']));
    const buf_69 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['enc.layers.3.norm2.bias']));
    const buf_70 = createEmptyBuf(device, 98304);;
    const buf_71 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.cls']));
    const buf_72 = createEmptyBuf(device, 98304);;
    const buf_73 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['rer.pos']));
    const buf_74 = createEmptyBuf(device, 294912);;
    const buf_75 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['rer.layers.0.self_attn.in_proj_weight']));
    const buf_76 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['rer.layers.0.self_attn.in_proj_bias']));
    const buf_77 = createEmptyBuf(device, 18432);;
    const buf_78 = createEmptyBuf(device, 6144);;
    const buf_79 = createEmptyBuf(device, 6144);;
    const buf_80 = createEmptyBuf(device, 98304);;
    const buf_81 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['rer.layers.0.self_attn.out_proj.weight']));
    const buf_82 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.self_attn.out_proj.bias']));
    const buf_83 = createEmptyBuf(device, 768);;
    const buf_84 = createEmptyBuf(device, 768);;
    const buf_85 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.norm1.weight']));
    const buf_86 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.norm1.bias']));
    const buf_87 = createEmptyBuf(device, 393216);;
    const buf_88 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.0.linear1.weight']));
    const buf_89 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['rer.layers.0.linear1.bias']));
    const buf_90 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.0.linear2.weight']));
    const buf_91 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.linear2.bias']));
    const buf_92 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.norm2.weight']));
    const buf_93 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.0.norm2.bias']));
    const buf_94 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['rer.layers.1.self_attn.in_proj_weight']));
    const buf_95 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['rer.layers.1.self_attn.in_proj_bias']));
    const buf_96 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['rer.layers.1.self_attn.out_proj.weight']));
    const buf_97 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.self_attn.out_proj.bias']));
    const buf_98 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.norm1.weight']));
    const buf_99 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.norm1.bias']));
    const buf_100 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.1.linear1.weight']));
    const buf_101 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['rer.layers.1.linear1.bias']));
    const buf_102 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.1.linear2.weight']));
    const buf_103 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.linear2.bias']));
    const buf_104 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.norm2.weight']));
    const buf_105 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.1.norm2.bias']));
    const buf_106 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['rer.layers.2.self_attn.in_proj_weight']));
    const buf_107 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['rer.layers.2.self_attn.in_proj_bias']));
    const buf_108 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['rer.layers.2.self_attn.out_proj.weight']));
    const buf_109 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.self_attn.out_proj.bias']));
    const buf_110 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.norm1.weight']));
    const buf_111 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.norm1.bias']));
    const buf_112 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.2.linear1.weight']));
    const buf_113 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['rer.layers.2.linear1.bias']));
    const buf_114 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.2.linear2.weight']));
    const buf_115 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.linear2.bias']));
    const buf_116 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.norm2.weight']));
    const buf_117 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.2.norm2.bias']));
    const buf_118 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['rer.layers.3.self_attn.in_proj_weight']));
    const buf_119 = createWeightBuf(device, 1536, getTensorBuffer(safetensor, metadata['rer.layers.3.self_attn.in_proj_bias']));
    const buf_120 = createWeightBuf(device, 65536, getTensorBuffer(safetensor, metadata['rer.layers.3.self_attn.out_proj.weight']));
    const buf_121 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.self_attn.out_proj.bias']));
    const buf_122 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.norm1.weight']));
    const buf_123 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.norm1.bias']));
    const buf_124 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.3.linear1.weight']));
    const buf_125 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['rer.layers.3.linear1.bias']));
    const buf_126 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['rer.layers.3.linear2.weight']));
    const buf_127 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.linear2.bias']));
    const output0 = createEmptyBuf(device, 256);;
    const buf_128 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.norm2.weight']));
    const buf_129 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.layers.3.norm2.bias']));
    const buf_130 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['rer.sc.weight']));
    const buf_131 = createWeightBuf(device, 4, getTensorBuffer(safetensor, metadata['rer.sc.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer1 = device.createBuffer({size:input1.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer2 = device.createBuffer({size:input2.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer3 = device.createBuffer({size:input3.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [E_53_2_4_16_4, E_2_212_2_8_16_4_4, r_53_8_16_3_4_32_4, r_424_8_8_16_3_4_32_4, r_53_53_8_4_4_16, r_4_53_53_16_8_4_4_16, r_53_8_4_53_4, r_848_32_4_53_4, r_53_8_4_53_4n1, r_848_32_4_53_4n1, r_53_8_4_4_4_53_4, r_16_53_4_8_4_4_4_53_4, r_53_2_16_4_4_8_16, r_8_53_2_8_16_4_4_8_16, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_4_4_32_4, r_424_8_8_16_4_4_32_4, r_53_2_16_4_4_128_4, r_424_2_8_16_4_4_128_4, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_3_4_32_4, r_424_8_8_16_3_4_32_4, r_53_53_8_4_4_16, r_4_53_53_16_8_4_4_16, r_53_8_4_53_4, r_848_32_4_53_4, r_53_8_4_53_4n1, r_848_32_4_53_4n1, r_53_8_4_4_4_53_4, r_16_53_4_8_4_4_4_53_4, r_53_2_16_4_4_8_16, r_8_53_2_8_16_4_4_8_16, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_4_4_32_4, r_424_8_8_16_4_4_32_4, r_53_2_16_4_4_128_4, r_424_2_8_16_4_4_128_4, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_3_4_32_4, r_424_8_8_16_3_4_32_4, r_53_53_8_4_4_16, r_4_53_53_16_8_4_4_16, r_53_8_4_53_4, r_848_32_4_53_4, r_53_8_4_53_4n1, r_848_32_4_53_4n1, r_53_8_4_4_4_53_4, r_16_53_4_8_4_4_4_53_4, r_53_2_16_4_4_8_16, r_8_53_2_8_16_4_4_8_16, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_4_4_32_4, r_424_8_8_16_4_4_32_4, r_53_2_16_4_4_128_4, r_424_2_8_16_4_4_128_4, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_3_4_32_4, r_424_8_8_16_3_4_32_4, r_53_53_8_4_4_16, r_4_53_53_16_8_4_4_16, r_53_8_4_53_4, r_848_32_4_53_4, r_53_8_4_53_4n1, r_848_32_4_53_4n1, r_53_8_4_4_4_53_4, r_16_53_4_8_4_4_4_53_4, r_53_2_16_4_4_8_16, r_8_53_2_8_16_4_4_8_16, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_424_2_8_16_4_4, r_53_8_16_4_4_32_4, r_424_8_8_16_4_4_32_4, r_53_2_16_4_4_128_4, r_424_2_8_16_4_4_128_4, r_212_16_8, r_106_32_4_32_4, r_212_16_8n1, r_106_32_4_32_4n1, E_53_2_16_4_4, E_16_4_8_16_3, E_16_16_8_3_4, r_8_8_8_16_3_3_32_4, r_4_16_8_3_3_16, r_16_32_3_3, r_16_32_3_3n1, r_16_4_8_4_4_3_3, r_8_2_8_16_4_3_8_16, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_4_3_32_4, r_8_2_8_16_4_3_128_4, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_3_3_32_4, r_4_16_8_3_3_16, r_16_32_3_3, r_16_32_3_3n1, r_16_4_8_4_4_3_3, r_8_2_8_16_4_3_8_16, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_4_3_32_4, r_8_2_8_16_4_3_128_4, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_3_3_32_4, r_4_16_8_3_3_16, r_16_32_3_3, r_16_32_3_3n1, r_16_4_8_4_4_3_3, r_8_2_8_16_4_3_8_16, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_4_3_32_4, r_8_2_8_16_4_3_128_4, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_3_3_32_4, r_4_16_8_3_3_16, r_16_32_3_3, r_16_32_3_3n1, r_16_4_8_4_4_3_3, r_8_2_8_16_4_3_8_16, r_192_16_8, r_192_16_8n1, E_8_2_8_16_4_3, r_8_8_8_16_4_3_32_4, r_8_2_8_16_4_3_128_4, r_192_16_8, r_192_16_8n1, r_64_16_8];
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

    return async (_input0,_input1,_input2,_input3) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
    await gpuWriteBuffer1.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer1.getMappedRange()).set(_input1);
        gpuWriteBuffer1.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer1, 0, input1, 0, gpuWriteBuffer1.size);
    await gpuWriteBuffer2.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer2.getMappedRange()).set(_input2);
        gpuWriteBuffer2.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer2, 0, input2, 0, gpuWriteBuffer2.size);
    await gpuWriteBuffer3.mapAsync(GPUMapMode.WRITE);
        new Int32Array(gpuWriteBuffer3.getMappedRange()).set(_input3);
        gpuWriteBuffer3.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer3, 0, input3, 0, gpuWriteBuffer3.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2, input1, buf_3], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_4, input2, buf_1, buf_2, input3, buf_3], [2, 212, 2]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_5, buf_0, buf_6, buf_7], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_8, buf_4, buf_6, buf_7], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_9, buf_5], [53, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_10, buf_8], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_11, buf_9], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_12, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_13, buf_9, buf_11], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_14, buf_10, buf_12], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_15, buf_9, buf_11, buf_13, buf_5], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_16, buf_10, buf_12, buf_14, buf_8], [53, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_17, buf_0, buf_15, buf_18, buf_19], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_20, buf_4, buf_16, buf_18, buf_19], [2, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_21, buf_17], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_22, buf_20], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_23, buf_17, buf_21], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_24, buf_20, buf_22], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_15, buf_17, buf_21, buf_23, buf_25, buf_26], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_16, buf_20, buf_22, buf_24, buf_25, buf_26], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_14, buf_15, buf_27, buf_28], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_29, buf_16, buf_27, buf_28], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_17, buf_15, buf_14, buf_30, buf_31], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_20, buf_16, buf_29, buf_30, buf_31], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_23, buf_17], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_24, buf_20], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_21, buf_17, buf_23], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_22, buf_20, buf_24], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_15, buf_17, buf_23, buf_21, buf_32, buf_33], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_16, buf_20, buf_24, buf_22, buf_32, buf_33], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_5, buf_15, buf_34, buf_35], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_8, buf_16, buf_34, buf_35], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_9, buf_5], [53, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_10, buf_8], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_13, buf_9], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_14, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_11, buf_9, buf_13], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_12, buf_10, buf_14], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_17, buf_9, buf_13, buf_11, buf_5], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_20, buf_10, buf_14, buf_12, buf_8], [53, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_0, buf_15, buf_17, buf_36, buf_37], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_4, buf_16, buf_20, buf_36, buf_37], [2, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_21, buf_0], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_22, buf_4], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_23, buf_0, buf_21], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_24, buf_4, buf_22], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_17, buf_0, buf_21, buf_23, buf_38, buf_39], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_20, buf_4, buf_22, buf_24, buf_38, buf_39], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_12, buf_17, buf_40, buf_41], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_29, buf_20, buf_40, buf_41], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_0, buf_17, buf_12, buf_42, buf_43], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_4, buf_20, buf_29, buf_42, buf_43], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_23, buf_0], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_24, buf_4], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_21, buf_0, buf_23], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_22, buf_4, buf_24], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_17, buf_0, buf_23, buf_21, buf_44, buf_45], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_20, buf_4, buf_24, buf_22, buf_44, buf_45], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_5, buf_17, buf_46, buf_47], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_8, buf_20, buf_46, buf_47], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_9, buf_5], [53, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_10, buf_8], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_11, buf_9], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_12, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_13, buf_9, buf_11], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_14, buf_10, buf_12], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_0, buf_9, buf_11, buf_13, buf_5], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_4, buf_10, buf_12, buf_14, buf_8], [53, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_15, buf_17, buf_0, buf_48, buf_49], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_16, buf_20, buf_4, buf_48, buf_49], [2, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_21, buf_15], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_22, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_23, buf_15, buf_21], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_24, buf_16, buf_22], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_0, buf_15, buf_21, buf_23, buf_50, buf_51], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_4, buf_16, buf_22, buf_24, buf_50, buf_51], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_14, buf_0, buf_52, buf_53], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_29, buf_4, buf_52, buf_53], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_15, buf_0, buf_14, buf_54, buf_55], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_16, buf_4, buf_29, buf_54, buf_55], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_23, buf_15], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_24, buf_16], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_21, buf_15, buf_23], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_22, buf_16, buf_24], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_0, buf_15, buf_23, buf_21, buf_56, buf_57], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_4, buf_16, buf_24, buf_22, buf_56, buf_57], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_5, buf_0, buf_58, buf_59], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_8, buf_4, buf_58, buf_59], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_9, buf_5], [53, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_10, buf_8], [53, 53, 4]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_13, buf_9], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_14, buf_10], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_11, buf_9, buf_13], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_12, buf_10, buf_14], [848, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_15, buf_9, buf_13, buf_11, buf_5], [53, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_16, buf_10, buf_14, buf_12, buf_8], [53, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_17, buf_0, buf_15, buf_60, buf_61], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_20, buf_4, buf_16, buf_60, buf_61], [2, 53, 8]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_21, buf_17], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_22, buf_20], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_23, buf_17, buf_21], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_24, buf_20, buf_22], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_15, buf_17, buf_21, buf_23, buf_62, buf_63], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_16, buf_20, buf_22, buf_24, buf_62, buf_63], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_12, buf_15, buf_64, buf_65], [8, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_29, buf_16, buf_64, buf_65], [8, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_17, buf_15, buf_12, buf_66, buf_67], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_20, buf_16, buf_29, buf_66, buf_67], [2, 424, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_23, buf_17], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_24, buf_20], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_21, buf_17, buf_23], [212, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_22, buf_20, buf_24], [106, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_15, buf_17, buf_23, buf_21, buf_68, buf_69], [2, 53, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_70, buf_71, buf_15, buf_20, buf_24, buf_22, buf_68, buf_69], [4, 16, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_72, buf_70, buf_73], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_74, buf_72, buf_75, buf_76], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_77, buf_74], [4, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_78, buf_77], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_79, buf_77, buf_78], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_70, buf_77, buf_78, buf_79, buf_74], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_80, buf_72, buf_70, buf_81, buf_82], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_83, buf_80], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_84, buf_80, buf_83], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_70, buf_80, buf_83, buf_84, buf_85, buf_86], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_87, buf_70, buf_88, buf_89], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_80, buf_70, buf_87, buf_90, buf_91], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_84, buf_80], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_83, buf_80, buf_84], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_70, buf_80, buf_84, buf_83, buf_92, buf_93], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_74, buf_70, buf_94, buf_95], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_77, buf_74], [4, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_79, buf_77], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_78, buf_77, buf_79], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_80, buf_77, buf_79, buf_78, buf_74], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_72, buf_70, buf_80, buf_96, buf_97], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_83, buf_72], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_84, buf_72, buf_83], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_80, buf_72, buf_83, buf_84, buf_98, buf_99], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_87, buf_80, buf_100, buf_101], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_72, buf_80, buf_87, buf_102, buf_103], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_84, buf_72], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_83, buf_72, buf_84], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_80, buf_72, buf_84, buf_83, buf_104, buf_105], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [buf_74, buf_80, buf_106, buf_107], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[144], layouts[144], infinityBuf, [buf_77, buf_74], [4, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[145], layouts[145], infinityBuf, [buf_78, buf_77], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[146], layouts[146], infinityBuf, [buf_79, buf_77, buf_78], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[147], layouts[147], infinityBuf, [buf_72, buf_77, buf_78, buf_79, buf_74], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[148], layouts[148], infinityBuf, [buf_70, buf_80, buf_72, buf_108, buf_109], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[149], layouts[149], infinityBuf, [buf_83, buf_70], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[150], layouts[150], infinityBuf, [buf_84, buf_70, buf_83], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[151], layouts[151], infinityBuf, [buf_72, buf_70, buf_83, buf_84, buf_110, buf_111], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[152], layouts[152], infinityBuf, [buf_87, buf_72, buf_112, buf_113], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[153], layouts[153], infinityBuf, [buf_70, buf_72, buf_87, buf_114, buf_115], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[154], layouts[154], infinityBuf, [buf_84, buf_70], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[155], layouts[155], infinityBuf, [buf_83, buf_70, buf_84], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[156], layouts[156], infinityBuf, [buf_72, buf_70, buf_84, buf_83, buf_116, buf_117], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[157], layouts[157], infinityBuf, [buf_74, buf_72, buf_118, buf_119], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[158], layouts[158], infinityBuf, [buf_77, buf_74], [4, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[159], layouts[159], infinityBuf, [buf_79, buf_77], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[160], layouts[160], infinityBuf, [buf_78, buf_77, buf_79], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[161], layouts[161], infinityBuf, [buf_70, buf_77, buf_79, buf_78, buf_74], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[162], layouts[162], infinityBuf, [buf_80, buf_72, buf_70, buf_120, buf_121], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[163], layouts[163], infinityBuf, [buf_83, buf_80], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[164], layouts[164], infinityBuf, [buf_84, buf_80, buf_83], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[165], layouts[165], infinityBuf, [buf_70, buf_80, buf_83, buf_84, buf_122, buf_123], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[166], layouts[166], infinityBuf, [buf_87, buf_70, buf_124, buf_125], [8, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[167], layouts[167], infinityBuf, [buf_80, buf_70, buf_87, buf_126, buf_127], [2, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[168], layouts[168], infinityBuf, [buf_84, buf_80], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[169], layouts[169], infinityBuf, [buf_83, buf_80, buf_84], [192, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[170], layouts[170], infinityBuf, [output0, buf_80, buf_84, buf_83, buf_128, buf_129, buf_130, buf_131], [64, 1, 1]);
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
