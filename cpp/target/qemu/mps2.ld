/* SPDX-License-Identifier: Apache-2.0
 *
 * QEMU's mps2-an386 memory map, for RUNNING the self-check binary.
 *
 * This is deliberately NOT `cpp/target/cortex-m4.ld`, and the two must not be confused.
 * That one carries the declared 32 KB / 8 KB budget so that a product-shaped image
 * overflowing it is a link error. This one is generous, because what it links is the test
 * harness -- 4,500-odd checks, printf, and newlib's semihosting stdio -- which is not the
 * product and was never meant to fit in the budget. Measuring size here would be measuring
 * the tests.
 *
 * The AN386 image puts ZBT SRAM for code at 0 and more SRAM at 0x20000000.
 */

MEMORY
{
  CODE (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(Reset_Handler)

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > CODE

  .text :
  {
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > CODE

  /* `__libc_init_array` calls `_init()` unconditionally, and `_init` lives in .init --
     contributed by crti.o and crtn.o. Leaving these unplaced does not produce a link
     error: the symbol resolves to zero and the call becomes a branch to address 0, which
     on ARMv7-M is an INVSTATE UsageFault because bit 0 is clear. That is exactly what
     happened, and it took a fault handler that printed CFSR and the stacked LR to see it
     rather than a lockup with PC=0 and no cause. */
  .init : { KEEP(*(SORT_NONE(.init))) } > CODE
  .fini : { KEEP(*(SORT_NONE(.fini))) } > CODE

  /* Unwind tables. -fno-exceptions does not remove them from newlib's own objects, and an
     unplaced allocatable section is a link error rather than something the linker discards. */
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx* .gnu.linkonce.armexidx.*); __exidx_end = .; } > CODE

  .preinit_array : { PROVIDE(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE(__preinit_array_end = .); } > CODE
  .init_array    : { PROVIDE(__init_array_start = .);    KEEP(*(SORT(.init_array*))); PROVIDE(__init_array_end = .); } > CODE
  .fini_array    : { PROVIDE(__fini_array_start = .);    KEEP(*(SORT(.fini_array*))); PROVIDE(__fini_array_end = .); } > CODE

  /* One load region and one run region, stated once. Giving both `AT (expr)` and `AT >` is
     a link error -- the first draft did, and said so plainly. */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    __data_start__ = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
    __data_end__ = .;
  } > SRAM AT > CODE
  _sidata = LOADADDR(.data);

  .bss (NOLOAD) :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = .;
  } > SRAM

  /* newlib's crt0 wants `__end__` to link at all, and this is where it comes from. But note
     what actually bounds the running image, because it is NOT these numbers and a review had
     to establish it: `rdimon-crt0.o` opens with a `SYS_HEAPINFO` semihosting call and, when
     the host answers, overwrites the stack pointer and its own `__heap_limit` with QEMU's
     values. Measured on this harness: the linked `_estack` is 0x20050398 and the live SP at
     `main()` is 0x21fffff0. So `_estack` is used for exactly one thing -- the initial MSP the
     hardware reads from the vector table before `Reset_Handler` runs -- and the 64 K and
     256 K below are a floor, not the bound. They are kept because the link needs the symbols
     and because a build without semihosting would then have somewhere to put them. */
  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    PROVIDE(end = .);
    PROVIDE(_end = .);
    PROVIDE(__end__ = .);
    . = . + 256K;
    PROVIDE(__HeapLimit = .);
  } > SRAM

  .stack (NOLOAD) :
  {
    . = ALIGN(8);
    . = . + 64K;
    _estack = .;
    PROVIDE(__StackTop = .);
    PROVIDE(__StackLimit = _estack - 64K);
  } > SRAM

  .ARM.attributes 0 : { *(.ARM.attributes) }
  /DISCARD/ : { *(.comment) }
}
