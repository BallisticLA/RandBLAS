.. _memory_tutorial:

Memory management 
=================

Here are the general rules for how sketching operators and sparse data matrices manange memory.

Allocation
  1. Our functions can allocate memory as needed.
      If memory is allocated, it must either be deallocated before the function returns
      or attached to an object that persists beyond the function's scope.
  2. Our functions may attach memory to RandBLAS-defined objects if:
      * The object has an own_memory member set to true.
      * The attachment involves redirecting a currently null pointer.
  3. Our functions must not attach memory to RandBLAS-defined objects if it risks causing a memory leak.

Deallocation
  1. We deallocate memory only in destructors. Consequently, we never reallocate memory.
      If reallocation is needed, the user must manage the deletion of old memory and either
      allocate new memory or set relevant pointers to null for RandBLAS to handle.
  2. A destructor attempts deallocation only if own_memory is true.
  3. If own_memory is true, a buffer is deleted if and only if it's non-null.

Assumptions and policies around non-null pointers
  1. RandBLAS trusts users to adhere to *documented* array length requirements when that length is const.
  2. During RandBLAS-managed allocation operations, we do not trust users to adhere to array length
      requirements if the length is determined by a non-const variable. As a result, some RandBLAS functions
      may raise errors when they detect non-null values for pointers that need to be redirected to new memory.
      This does not remove the user's control over managing pointer members and the own_memory member.
