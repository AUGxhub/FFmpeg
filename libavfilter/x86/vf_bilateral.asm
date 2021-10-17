;*****************************************************************************
;* x86-optimized functions for bilateral filter
;*
;* Copyright (c) 2021 Paul B Mahol
;*
;* This file is part of FFmpeg.
;*
;* FFmpeg is free software; you can redistribute it and/or
;* modify it under the terms of the GNU Lesser General Public
;* License as published by the Free Software Foundation; either
;* version 2.1 of the License, or (at your option) any later version.
;*
;* FFmpeg is distributed in the hope that it will be useful,
;* but WITHOUT ANY WARRANTY; without even the implied warranty of
;* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;* Lesser General Public License for more details.
;*
;* You should have received a copy of the GNU Lesser General Public
;* License along with FFmpeg; if not, write to the Free Software
;* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
;******************************************************************************

%include "libavutil/x86/x86util.asm"

SECTION .text

INIT_YMM avx
cglobal sum_and_div_float, 6,6,6, dst, num0, num1, den0, den1, w
    movsxdifnidn wq, wd

    sal          wd, 2
    add        dstq, wq
    add       num0q, wq
    add       num1q, wq
    add       den0q, wq
    add       den1q, wq
    neg          wq

    .loop:
        movaps                m0, [num0q + wq]
        movaps                m2, [num0q + wq + mmsize]
        movaps                m1, [den0q + wq]
        movaps                m3, [den0q + wq + mmsize]
        addps                 m1, [den1q + wq]
        addps                 m3, [den1q + wq + mmsize]
        rcpps                 m1, m1
        rcpps                 m3, m3

        addps                 m0, [num1q + wq]
        addps                 m2, [num1q + wq + mmsize]
        mulps                 m0, m1
        mulps                 m2, m3

        movaps                [dstq + wq], m0
        movaps                [dstq + wq + mmsize], m2

        add                   wq, mmsize * 2
        jl .loop
    RET
