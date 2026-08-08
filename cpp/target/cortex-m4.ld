/* SPDX-License-Identifier: Apache-2.0
 *
 * A deliberately generic Cortex-M4F memory map, for measuring THIS LIBRARY'S size and
 * nothing else.
 *
 * It is not a board support file and must not be mistaken for one: there is no vector
 * table, no startup code, no clock setup and no peripheral regions. Its only job is to
 * give the linker somewhere to put sections so that a map file can be produced, and so
 * that `arm-none-eabi-size` reports a figure that came from a LINK rather than from the
 * sum of some object files. DESIGN.md requires that distinction.
 *
 * The sizes below are the declared budget from DESIGN.md -- 32 KB flash, 8 KB RAM -- so
 * that overflowing the budget is a link error rather than a number somebody has to notice
 * in a report.
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 32K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 8K
}

ENTRY(_am_entry)

SECTIONS
{
  .text :
  {
    KEEP(*(.am_entry))
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
    _sidata = .;
  } > FLASH

  .data : AT (_sidata)
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  /* Anything the toolchain emits that this script has not placed is a surprise, and a
     surprise in a size measurement is a wrong measurement. Fail rather than discard. */
  .ARM.attributes 0 : { *(.ARM.attributes) }
  /DISCARD/ : { *(.comment) }
}
