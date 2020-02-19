//
//  Shaders.metal
//  GMetal Shared
//
//  Created by LEE CHUL HYUN on 2/16/20.
//  Copyright © 2020 LEE CHUL HYUN. All rights reserved.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal [[attribute(VertexAttributeNormal)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float3 normal;
    float2 texCoord;
    float3 worldNormal;
    float3 worldPosition;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.normal = in.normal;
    out.texCoord = in.texCoord;
    out.worldPosition = (uniforms.modelViewMatrix * position).xyz;
    out.worldNormal = uniforms.normalMatrix * in.normal;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
    float3 baseColor = float3(0.5, 0.6, 0.7);
    float3 lightColor1 = float3(1,1,1);
    float3 lightColor2 = float3(1,0,0);
    float3 lightColor3 = float3(0,1,0);
    float3 lightColor4 = float3(0,0,1);
    float3 lightPosition1 = normalize(float3(0,1,-1));
    float3 lightPosition2 = normalize(float3(1,0,-1));
    float3 lightPosition3 = normalize(float3(0,-1,-1));
    float3 lightPosition4 = normalize(float3(-1,0,-1));
    float3 normalDirection = normalize(in.worldNormal);
    float3 ambientColor = float3(0.5, 0.5, 0.5) * 0.1;
    float3 diffuseColor1 = saturate(dot(lightPosition1, normalDirection)) * lightColor1 * baseColor;
    float3 diffuseColor2 = saturate(dot(lightPosition2, normalDirection)) * lightColor2 * baseColor;
    float3 diffuseColor3 = saturate(dot(lightPosition3, normalDirection)) * lightColor3 * baseColor;
    float3 diffuseColor4 = saturate(dot(lightPosition4, normalDirection)) * lightColor4 * baseColor;
    float3 color = ambientColor + diffuseColor1 + diffuseColor2 + diffuseColor3 + diffuseColor4;
//    float3 color = diffuseColor4;
    return float4(color, 1);
//    return float4(colorSample);
//    return float4(in.normal.x >= 0 ? in.normal.x : -in.normal.x,
//                  in.normal.y >= 0 ? in.normal.y : -in.normal.y,
//                  in.normal.z >= 0 ? in.normal.z : -in.normal.z,  1);

//    return float4(1.0, 0, 0, 1);
}

vertex ColorInOut vertexShader2(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;
    
    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.normal = in.normal;
    out.texCoord = in.texCoord;
    out.worldPosition = (uniforms.modelViewMatrix * position).xyz;
    out.worldNormal = uniforms.normalMatrix * in.normal;
    
    return out;
}

fragment float4 fragmentShader2(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]])
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);
    
    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
//    float3 baseColor = float3(in.normal.x >= 0 ? in.normal.x : -in.normal.x,
//                              in.normal.y >= 0 ? in.normal.y : -in.normal.y,
//                              in.normal.z >= 0 ? in.normal.z : -in.normal.z);
    float3 lightColor1 = float3(1,1,1);
    float3 lightColor2 = float3(1,0,0);
    float3 lightColor3 = float3(0,1,0);
    float3 lightColor4 = float3(0,0,1);
    float3 lightPosition1 = normalize(float3(0,1,-1));
    float3 lightPosition2 = normalize(float3(1,0,-1));
    float3 lightPosition3 = normalize(float3(0,-1,-1));
    float3 lightPosition4 = normalize(float3(-1,0,-1));
    float3 normalDirection = normalize(in.worldNormal);
    float3 baseColor = float3(normalDirection.x, normalDirection.y, -normalDirection.z);
    float3 ambientColor = float3(0.5, 0.5, 0.5) * 0.1;
    float3 diffuseColor1 = saturate(dot(lightPosition1, normalDirection)) * lightColor1 * baseColor;
    float3 diffuseColor2 = saturate(dot(lightPosition2, normalDirection)) * lightColor2 * baseColor;
    float3 diffuseColor3 = saturate(dot(lightPosition3, normalDirection)) * lightColor3 * baseColor;
    float3 diffuseColor4 = saturate(dot(lightPosition4, normalDirection)) * lightColor4 * baseColor;
    float3 color = ambientColor + diffuseColor1 + diffuseColor2 + diffuseColor3 + diffuseColor4;
    //    float3 color = diffuseColor4;
    return float4(color, 1);
    //    return float4(colorSample);
//    return float4(in.normal.x >= 0 ? in.normal.x : -in.normal.x,
//                  in.normal.y >= 0 ? in.normal.y : -in.normal.y,
//                  in.normal.z >= 0 ? in.normal.z : -in.normal.z,  1);
    
//        return float4(1.0, 0, 0, 1);
}
