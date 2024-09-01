.. _memory_tutorial:

Memory management 
=================

Decades ago, the designers of the classic BLAS made the wise decision to not internally allocate dynamically-sized
arrays.
Such an approach was (and is) viabale because BLAS only operates on very simple datatypes: scalars and references
to dense arrays of scalars.

RandBLAS, by contrast, needs to provide a polymorphic API with far more sophisticated datatypes.
This has led us to adopt a policy where we can internally allocate and dellocate dynamically-sized arrays 
with the ``new []`` and ``delete []`` keywords, subject to the restrictions below. 
Users are not bound to follow these rules, but deviations from them should be made with care.

Allocation and writing to reference members
-------------------------------------------

1. We allocate memory with ``new []`` only when necessary or explicitly requested.
    
    If a region of memory is allocated, it must either be deallocated before the function returns
    or attached to a RandBLAS-defined object that persists beyond the function's scope.

2. We can only attach memory to objects by overwriting a null-valued reference member,
   and only when the object has an ``own_memory`` member that evaluates to true.

3. We cannot overwrite an an object's reference member if there is a chance that doing so may cause a memory leak.
    
    This restriction is in place regardless of whether ``obj.own_memory`` is true.
    It makes for very few cases when RandBLAS is allowed to overwrite a non-null reference member.

Deallocation
------------

1. We deallocate memory only in destructors.

    In particular, we never "reallocate" memory. If reallocation is needed, the user must manage the deletion of old memory
    and then put the object in a state where RandBLAS can write to it.

2. A destructor attempts deallocation only if ``own_memory`` is true.

    The destructor calls ``delete []`` on a specific reference member if and only if that member is non-null.

What we do instead of overwriting non-null references 
-----------------------------------------------------

Let ``obj`` denote an instance of a RandBLAS-defined type where  ``obj.member`` is a reference.
Suppose we find ourselves in a situation where ``obj.member`` is *non-null*,
but we're at a point in RandBLAS' code that would have written to ``obj.member`` if it were null.
There are two possibilities for what happens next.

1. If the documentation for ``obj.member`` states an array length requirement purely in terms of ``const`` members,
   then we silently skip memory allocations that would overwrite ``obj.member``. We'll simply
   assume that ``obj.member`` has the correct size.

2. Under any other circumstances, RandBLAS raises an error. 

In essence, the first situation has enough structure that the user could plausibly understand RandBLAS' behavior,
while the latter situation is too error prone for RandBLAS to play a role in it.


Discussion
----------

Clarifications:
 * No RandBLAS datatypes use ``new`` or ``new []`` in their constructors.
   Reference members of such datatypes are either null-initialized or initialized at user-provided values.
 * Move-constructors are allowed to overwrite an object's reference members with ``nullptr`` if those references have been copied to
   a newly-created object.
 * Users retain the ability to overwrite any RandBLAS object's ``own_member`` member and its reference members at any time;
   no such members are declared as const.

We're not totally satisfied with this document writ-large.
It would probably be better if removed the commentary from the enumerations above and added lots of examples that refer to actual RandBLAS code.
Alas, this will have to do for now.
Questions about what specific parts of this policy mean or proposed revisions are welcome!
Please get in touch with us on GitHub.
