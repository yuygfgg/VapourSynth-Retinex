/*
* Retinex filter - VapourSynth plugin
* Copyright (C) 2014  mawen1250
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "Gaussian.h"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Recursive_Gaussian_Parameters(const double sigma, FLType & B, FLType & B1, FLType & B2, FLType & B3)
{
    const double q = sigma < 2.5 ? 3.97156 - 4.14554*sqrt(1 - 0.26891*sigma) : 0.98711*sigma - 0.96330;

    const double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    const double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    const double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    const double b3 = 0.422205*q*q*q;

    B = static_cast<FLType>(1 - (b1 + b2 + b3) / b0);
    B1 = static_cast<FLType>(b1 / b0);
    B2 = static_cast<FLType>(b2 / b0);
    B3 = static_cast<FLType>(b3 / b0);
}

#ifdef __ARM_NEON__

constexpr int vectorSize = 2;

void Recursive_Gaussian2D_Vertical(FLType * output, const FLType * input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    
    float64x2_t vB = vdupq_n_f64(B);
    float64x2_t vB1 = vdupq_n_f64(B1);
    float64x2_t vB2 = vdupq_n_f64(B2);
    float64x2_t vB3 = vdupq_n_f64(B3);

    if (output != input)
    {
        memcpy(output, input, sizeof(FLType) * width);
    }

    int vectorWidth = width / vectorSize * vectorSize;
    
    for (int j = 0; j < height; j++)
    {
        int lower = stride * j;
        
        for (int i = 0; i < vectorWidth; i += vectorSize)
        {
            int curr = lower + i;
            int prev1 = j < 1 ? curr : curr - stride;
            int prev2 = j < 2 ? prev1 : prev1 - stride;
            int prev3 = j < 3 ? prev2 : prev2 - stride;

            float64x2_t vP0 = vld1q_f64(&input[curr]);
            float64x2_t vP1 = vld1q_f64(&output[prev1]);
            float64x2_t vP2 = vld1q_f64(&output[prev2]);
            float64x2_t vP3 = vld1q_f64(&output[prev3]);

            float64x2_t vResult = vmulq_f64(vB, vP0);
            vResult = vfmaq_f64(vResult, vB1, vP1);
            vResult = vfmaq_f64(vResult, vB2, vP2);
            vResult = vfmaq_f64(vResult, vB3, vP3);

            vst1q_f64(&output[curr], vResult);
        }

        for (int i = vectorWidth; i < width; i++)
        {
            int curr = lower + i;
            int prev1 = j < 1 ? curr : curr - stride;
            int prev2 = j < 2 ? prev1 : prev1 - stride;
            int prev3 = j < 3 ? prev2 : prev2 - stride;

            FLType P0 = input[curr];
            FLType P1 = output[prev1];
            FLType P2 = output[prev2];
            FLType P3 = output[prev3];
            output[curr] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }

    for (int j = height - 1; j >= 0; j--)
    {
        int lower = stride * j;

        for (int i = 0; i < vectorWidth; i += vectorSize)
        {
            int curr = lower + i;
            int next1 = j >= height - 1 ? curr : curr + stride;
            int next2 = j >= height - 2 ? next1 : next1 + stride;
            int next3 = j >= height - 3 ? next2 : next2 + stride;

            float64x2_t vP0 = vld1q_f64(&output[curr]);
            float64x2_t vP1 = vld1q_f64(&output[next1]);
            float64x2_t vP2 = vld1q_f64(&output[next2]);
            float64x2_t vP3 = vld1q_f64(&output[next3]);

            float64x2_t vResult = vmulq_f64(vB, vP0);
            vResult = vfmaq_f64(vResult, vB1, vP1);
            vResult = vfmaq_f64(vResult, vB2, vP2);
            vResult = vfmaq_f64(vResult, vB3, vP3);

            vst1q_f64(&output[curr], vResult);
        }

        for (int i = vectorWidth; i < width; i++)
        {
            int curr = lower + i;
            int next1 = j >= height - 1 ? curr : curr + stride;
            int next2 = j >= height - 2 ? next1 : next1 + stride;
            int next3 = j >= height - 3 ? next2 : next2 + stride;

            FLType P0 = output[curr];
            FLType P1 = output[next1];
            FLType P2 = output[next2];
            FLType P3 = output[next3];
            output[curr] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }
}

void Recursive_Gaussian2D_Horizontal(FLType * output, const FLType * input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    float64x2_t vB = vdupq_n_f64(B);
    float64x2_t vB1 = vdupq_n_f64(B1);
    float64x2_t vB2 = vdupq_n_f64(B2);
    float64x2_t vB3 = vdupq_n_f64(B3);

    for (int j = 0; j < height; j++)
    {
        int lower = stride * j;
        int upper = lower + width;

        output[lower] = input[lower];
        FLType P1 = output[lower];
        FLType P2 = P1;
        FLType P3 = P1;

        int i = lower + 1;
        for (; i < lower + 3 && i < upper; i++)
        {
            FLType P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        float64x2_t vP1 = vdupq_n_f64(P1);
        float64x2_t vP2 = vdupq_n_f64(P2);
        float64x2_t vP3 = vdupq_n_f64(P3);

        for (; i + vectorSize <= upper; i += vectorSize)
        {
            float64x2_t vIn = vld1q_f64(&input[i]);
            float64x2_t vResult = vmulq_f64(vB, vIn);
            vResult = vfmaq_f64(vResult, vB1, vP1);
            vResult = vfmaq_f64(vResult, vB2, vP2);
            vResult = vfmaq_f64(vResult, vB3, vP3);

            vP3 = vP2;
            vP2 = vP1;
            vP1 = vResult;
            vst1q_f64(&output[i], vResult);
        }

        P1 = vgetq_lane_f64(vP1, 1);
        P2 = vgetq_lane_f64(vP2, 1);
        P3 = vgetq_lane_f64(vP3, 1);

        for (; i < upper; i++)
        {
            FLType P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        i = upper - 1;
        P1 = P2 = P3 = output[i];

        i--;
        for (; i > upper - 4 && i >= lower; i--)
        {
            FLType P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        vP1 = vdupq_n_f64(P1);
        vP2 = vdupq_n_f64(P2);
        vP3 = vdupq_n_f64(P3);

        for (; i >= lower + vectorSize; i -= vectorSize)
        {
            float64x2_t vOut = vld1q_f64(&output[i-1]);
            float64x2_t vResult = vmulq_f64(vB, vOut);
            vResult = vfmaq_f64(vResult, vB1, vP1);
            vResult = vfmaq_f64(vResult, vB2, vP2);
            vResult = vfmaq_f64(vResult, vB3, vP3);

            vP3 = vP2;
            vP2 = vP1;
            vP1 = vResult;
            vst1q_f64(&output[i-1], vResult);
        }

        P1 = vgetq_lane_f64(vP1, 0);
        P2 = vgetq_lane_f64(vP2, 0);
        P3 = vgetq_lane_f64(vP3, 0);

        for (; i >= lower; i--)
        {
            FLType P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}

#else

void Recursive_Gaussian2D_Vertical(FLType * output, const FLType * input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    int i0, i1, i2, i3, j, lower, upper;
    FLType P0, P1, P2, P3;

    if (output != input)
    {
        memcpy(output, input, sizeof(FLType) * width);
    }

    for (j = 0; j < height; j++)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j < 1 ? i0 : i0 - stride;
        i2 = j < 2 ? i1 : i1 - stride;
        i3 = j < 3 ? i2 : i2 - stride;

        for (; i0 < upper; i0++, i1++, i2++, i3++)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = input[i0];
            output[i0] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }

    for (j = height - 1; j >= 0; j--)
    {
        lower = stride * j;
        upper = lower + width;

        i0 = lower;
        i1 = j >= height - 1 ? i0 : i0 + stride;
        i2 = j >= height - 2 ? i1 : i1 + stride;
        i3 = j >= height - 3 ? i2 : i2 + stride;

        for (; i0 < upper; i0++, i1++, i2++, i3++)
        {
            P3 = output[i3];
            P2 = output[i2];
            P1 = output[i1];
            P0 = output[i0];
            output[i0] = B*P0 + B1*P1 + B2*P2 + B3*P3;
        }
    }
}

void Recursive_Gaussian2D_Horizontal(FLType * output, const FLType * input, int height, int width, int stride, const FLType B, const FLType B1, const FLType B2, const FLType B3)
{
    int i, j, lower, upper;
    FLType P0, P1, P2, P3;

    for (j = 0; j < height; j++)
    {
        lower = stride * j;
        upper = lower + width;

        i = lower;
        output[i] = P3 = P2 = P1 = input[i];

        for (i++; i < upper; i++)
        {
            P0 = B*input[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }

        i--;
        P3 = P2 = P1 = output[i];

        for (i--; i >= lower; i--)
        {
            P0 = B*output[i] + B1*P1 + B2*P2 + B3*P3;
            P3 = P2;
            P2 = P1;
            P1 = P0;
            output[i] = P0;
        }
    }
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
