.global matmul_block_4x8_avx2_asm
.type matmul_block_4x8_avx2_asm, @function
.intel_syntax noprefix
.text

# void matmul_block_4x8_avx2_asm(
#     float* out_ptr,       // rdi
#     const float* lhs_ptr,  // rsi
#     const float* rhs_ptr,  // rdx
#     size_t k_l,            // rcx
#     size_t n               // r8
# )
matmul_block_4x8_avx2_asm:
    push r12
    push r13
    push r14
    push r15

    # Initialize ptr0 = lhs_ptr
    mov r10, rsi

    # r11 = k_l * 4 (stride of A in bytes)
    mov r11, rcx
    shl r11, 2

    # Initialize ptr1, ptr2, ptr3
    mov r12, r10
    add r12, r11            # ptr1 = ptr0 + k_l * 4
    
    mov r13, r12
    add r13, r11            # ptr2 = ptr1 + k_l * 4
    
    mov r14, r13
    add r14, r11            # ptr3 = ptr2 + k_l * 4

    # r9 = n * 4 (stride of B in bytes)
    mov r9, r8
    shl r9, 2

    # Clear accumulators
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3

    # r15 = r_k = 0
    xor r15, r15

.Linner_loop:
    cmp r15, rcx
    jae .Linner_loop_end

    # Load B row (8 floats)
    vmovups ymm4, [rdx]

    # Broadcast A elements
    vbroadcastss ymm5, [r10]
    vbroadcastss ymm6, [r12]
    vbroadcastss ymm7, [r13]
    vbroadcastss ymm8, [r14]

    # FMA
    vfmadd231ps ymm0, ymm5, ymm4
    vfmadd231ps ymm1, ymm6, ymm4
    vfmadd231ps ymm2, ymm7, ymm4
    vfmadd231ps ymm3, ymm8, ymm4

    # Increment A pointers
    add r10, 4
    add r12, 4
    add r13, 4
    add r14, 4

    # Increment B pointer by stride (n * 4 bytes)
    add rdx, r9

    inc r15
    jmp .Linner_loop

.Linner_loop_end:
    # Store accumulators back to C
    # c_stride = n * 4
    vmovups [rdi], ymm0
    
    add rdi, r9
    vmovups [rdi], ymm1
    
    add rdi, r9
    vmovups [rdi], ymm2
    
    add rdi, r9
    vmovups [rdi], ymm3

    pop r15
    pop r14
    pop r13
    pop r12
    ret
