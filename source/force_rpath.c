/**********************************************************************
  force_rpath.c

  Rewrites the DT_RUNPATH dynamic tag of an ELF binary into DT_RPATH.

  The nvc link driver appends --enable-new-dtags after every
  user-supplied linker flag, so the library search path recorded in
  the openmx binary always becomes DT_RUNPATH, which LD_LIBRARY_PATH
  overrides.  A stray CUDA installation early in LD_LIBRARY_PATH
  (e.g. /usr/local/cuda/lib64) then replaces the cuBLAS/cuSOLVER
  generation the binary was linked against, which corrupts the GPU
  solvers with asynchronous illegal-address faults.  DT_RPATH takes
  precedence over LD_LIBRARY_PATH, so flipping the tag pins the
  binary to the libraries it was built with; search directories that
  do not exist at run time are simply skipped by the loader, and
  LD_PRELOAD still works for deliberate overrides.

  Usage: force_rpath <elf64-binary>
***********************************************************************/

#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  FILE *fp;
  Elf64_Ehdr eh;
  Elf64_Phdr ph;
  Elf64_Dyn dyn;
  long dyn_off;
  unsigned long dyn_size, i, ndyn;
  int flipped, has_rpath;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <elf64-binary>\n", argv[0]);
    return 1;
  }

  fp = fopen(argv[1], "r+b");
  if (fp == NULL) {
    fprintf(stderr, "force_rpath: cannot open %s\n", argv[1]);
    return 1;
  }

  if (fread(&eh, sizeof(eh), 1, fp) != 1 ||
      memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 ||
      eh.e_ident[EI_CLASS] != ELFCLASS64 ||
      eh.e_ident[EI_DATA] != ELFDATA2LSB) {
    fprintf(stderr, "force_rpath: %s is not a little-endian ELF64 file\n", argv[1]);
    fclose(fp);
    return 1;
  }

  /* locate the PT_DYNAMIC segment */

  dyn_off = -1;
  dyn_size = 0;
  for (i = 0; i < eh.e_phnum; i++) {
    if (fseek(fp, (long)(eh.e_phoff + i*eh.e_phentsize), SEEK_SET) != 0 ||
        fread(&ph, sizeof(ph), 1, fp) != 1) {
      fprintf(stderr, "force_rpath: cannot read the program headers of %s\n", argv[1]);
      fclose(fp);
      return 1;
    }
    if (ph.p_type == PT_DYNAMIC) {
      dyn_off = (long)ph.p_offset;
      dyn_size = (unsigned long)ph.p_filesz;
      break;
    }
  }

  if (dyn_off < 0) {
    printf("force_rpath: %s has no dynamic segment, nothing to do\n", argv[1]);
    fclose(fp);
    return 0;
  }

  /* flip every DT_RUNPATH entry into DT_RPATH */

  flipped = 0;
  has_rpath = 0;
  ndyn = dyn_size/sizeof(Elf64_Dyn);
  for (i = 0; i < ndyn; i++) {
    if (fseek(fp, dyn_off + (long)(i*sizeof(Elf64_Dyn)), SEEK_SET) != 0 ||
        fread(&dyn, sizeof(dyn), 1, fp) != 1) {
      fprintf(stderr, "force_rpath: cannot read the dynamic section of %s\n", argv[1]);
      fclose(fp);
      return 1;
    }
    if (dyn.d_tag == DT_NULL) break;
    if (dyn.d_tag == DT_RPATH) has_rpath = 1;
    if (dyn.d_tag == DT_RUNPATH) {
      dyn.d_tag = DT_RPATH;
      if (fseek(fp, dyn_off + (long)(i*sizeof(Elf64_Dyn)), SEEK_SET) != 0 ||
          fwrite(&dyn, sizeof(dyn), 1, fp) != 1) {
        fprintf(stderr, "force_rpath: cannot rewrite the dynamic section of %s\n", argv[1]);
        fclose(fp);
        return 1;
      }
      flipped++;
    }
  }

  fclose(fp);

  if (flipped)
    printf("force_rpath: %s: DT_RUNPATH -> DT_RPATH (library search path now beats LD_LIBRARY_PATH)\n",
           argv[1]);
  else if (has_rpath)
    printf("force_rpath: %s: already DT_RPATH, nothing to do\n", argv[1]);
  else
    printf("force_rpath: %s: no DT_RUNPATH recorded, nothing to do\n", argv[1]);

  return 0;
}
