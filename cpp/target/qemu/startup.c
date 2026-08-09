/* SPDX-License-Identifier: Apache-2.0
 *
 * Just enough Cortex-M4 to run the self-checks under QEMU's mps2-an386.
 *
 * THIS COULD NEVER BE FIRMWARE, and the reason is stronger than the usual "no clock setup, no
 * drivers". The image is linked against `--specs=rdimon.specs`, so its stdio and its heap
 * come from semihosting: real silicon with no debug host attached faults on the first
 * `bkpt 0xAB`. `cpp/target/cortex-m4.ld` and `cpp/target/usage.c` are the product-shaped
 * pair; this one exists to run tests and nothing else.
 *
 * Three jobs and no more.
 *
 * The vector table gives the core an initial stack pointer and a reset entry, because that
 * is how ARMv7-M boots -- it reads both from address 0 rather than jumping to a symbol.
 *
 * The reset handler turns the FPU on before anything else runs: CPACR's CP10/CP11 fields
 * come up as "access denied", so the first VFP instruction in a hard-float binary faults.
 * That is the classic way a cross build that compiles perfectly dies on its first
 * instruction, and it is worth stating rather than leaving as three magic lines.
 *
 * And the fault handler REPORTS -- through raw semihosting, not printf.
 *
 * Both of those took a correction. The first version had a two-entry vector table, so a
 * UsageFault had nowhere to go: the core read a null handler, faulted again, and locked up
 * with PC=0 and the fault cause already lost. The second version had sixteen entries and
 * called `printf` from the handler, which locked up in a subtler way -- the original fault
 * was *inside* printf, so re-entering stdio from the handler faulted again. A fault
 * reporter may not depend on anything that can be the thing that broke. `bkpt 0xAB` with
 * SYS_WRITE0 depends on nothing but the debug channel.
 */

extern unsigned _estack;
extern void _start(void);

void Reset_Handler(void);
void Default_Handler(void);
void am_report_fault(unsigned* frame);

/* -- raw semihosting, so the reporter shares no state with the program ----------------- */

/* The operation code is moved into r0 inside the asm rather than through a local register
   variable. GCC only guarantees a `register ... __asm__("r0")` binding for the asm
   immediately following it, and at -Os it did not hold: the first call worked and the
   second reached `bkpt` with r0 still holding a leftover, which QEMU reported as an
   unsupported semihosting call. Doing the moves explicitly removes the guarantee from the
   equation. */
static void sh_write0(const char* s) {
    __asm__ volatile(
        "mov r0, #4\n"          /* SYS_WRITE0 */
        "mov r1, %0\n"
        "bkpt 0xab\n"
        :
        : "r"(s)
        : "r0", "r1", "memory");
}

__attribute__((noreturn)) static void sh_exit(void) {
    __asm__ volatile(
        "mov  r0, #0x18\n"       /* SYS_EXIT                        */
        "ldr  r1, =0x20023\n"    /* ADP_Stopped_RunTimeErrorUnknown */
        "bkpt 0xab\n"
        :
        :
        : "r0", "r1", "memory");
    for (;;) {
    }
}

static void sh_hex(const char* label, unsigned v) {
    static const char digits[] = "0123456789abcdef";
    char buf[16];
    for (int i = 0; i < 8; ++i) buf[i] = digits[(v >> (28 - 4 * i)) & 0xF];
    buf[8] = '\n';
    buf[9] = 0;
    sh_write0(label);
    sh_write0(buf);
}

/* -- the handler ----------------------------------------------------------------------- */

__attribute__((naked)) void Default_Handler(void) {
    /* Pick the stack the exception frame is on -- EXC_RETURN bit 2 says which -- and hand
       the frame to the reporter. Naked, because a prologue would move the frame. */
    __asm__ volatile(
        "tst   lr, #4          \n"
        "ite   eq              \n"
        "mrseq r0, msp         \n"
        "mrsne r0, psp         \n"
        "b     am_report_fault \n");
}

void am_report_fault(unsigned* frame) {
    sh_write0("\n*** FAULT ***\n");
    sh_hex("  cfsr  0x", *(volatile unsigned*)0xE000ED28u);
    sh_hex("  hfsr  0x", *(volatile unsigned*)0xE000ED2Cu);
    sh_hex("  bfar  0x", *(volatile unsigned*)0xE000ED38u);
    sh_hex("  pc    0x", frame[6]);
    sh_hex("  lr    0x", frame[5]);
    sh_hex("  xpsr  0x", frame[7]);
    sh_exit();
}

/* Sixteen entries: the architecturally defined system exceptions. Everything but reset
   lands in the reporter. */
__attribute__((section(".isr_vector"), used))
void* const vector_table[16] = {
    (void*)&_estack,        /*  0 initial SP        */
    (void*)Reset_Handler,   /*  1 reset             */
    (void*)Default_Handler, /*  2 NMI               */
    (void*)Default_Handler, /*  3 HardFault         */
    (void*)Default_Handler, /*  4 MemManage         */
    (void*)Default_Handler, /*  5 BusFault          */
    (void*)Default_Handler, /*  6 UsageFault        */
    0, 0, 0, 0,             /*  7-10 reserved       */
    (void*)Default_Handler, /* 11 SVCall            */
    (void*)Default_Handler, /* 12 DebugMonitor      */
    0,                      /* 13 reserved          */
    (void*)Default_Handler, /* 14 PendSV            */
    (void*)Default_Handler, /* 15 SysTick           */
};

void Reset_Handler(void) {
    /* CPACR: full access to CP10 and CP11, which is the single-precision FPU the declared
       floor requires. Without this the hard-float ABI's very first VFP instruction traps. */
    volatile unsigned* const cpacr = (volatile unsigned*)0xE000ED88u;
    *cpacr |= (0xFu << 20);
    __asm__ volatile("dsb; isb");

    /* Report the separate fault kinds rather than escalating everything to HardFault, so
       CFSR says something useful. SHCSR: USGFAULTENA | BUSFAULTENA | MEMFAULTENA. */
    volatile unsigned* const shcsr = (volatile unsigned*)0xE000ED24u;
    *shcsr |= (1u << 18) | (1u << 17) | (1u << 16);

    /* Copy .data from its load address to its run address.
     *
     * newlib's crt0 zeroes .bss and does NOT do this -- it assumes an operating system's
     * loader already placed .data at its run address, which is true for a hosted program
     * and false for a ROM image whose LMA and VMA differ. Nothing warns about it. What
     * happens instead is that .data reads as whatever the RAM came up as, and the first
     * casualty is newlib's own `_impure_ptr` and the initial FILE structures: the third
     * fault this file had to diagnose was `__sflush_r` doing `blx r6` on a garbage write
     * pointer, which is stdout's vtable read out of uninitialised memory.
     */
    extern unsigned _sidata, _sdata, _edata;
    const unsigned* src = &_sidata;
    unsigned* dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;

    _start();
    for (;;) {
    }
}
