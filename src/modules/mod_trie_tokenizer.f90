! Copyright 2023 FortAI-Hub contributors.
! Released under the MIT License. See LICENSE file for full license information.

module mod_trie_tokenizer
    use iso_fortran_env, only : int16
    implicit none

    private
    public :: load_trie_tokenizer, trie_tokenizer, token

    type :: trie_p
        type(trie), pointer :: p => null()
    end type trie_p

    type :: trie
        character :: char
        integer :: index
        type(trie_p) :: next(256)
    contains
        procedure :: add => trie_add
        procedure :: print => trie_print
        procedure :: find_longest => trie_find_longest
    end type trie

    type :: token
        character(:), allocatable :: value
    end type token

    type :: trie_tokenizer
        type(token), allocatable :: tokens(:)
        type(trie) :: root
    contains
        procedure :: encode => trie_tokenizer_encode
        procedure :: decode => trie_tokenizer_decode
    end type trie_tokenizer

contains

    function load_trie_tokenizer(filename) result(tokenizer)
        character(*), intent(in) :: filename
        type(trie_tokenizer) :: tokenizer

        integer :: fileunit, tokens_count, token_index, length
        integer(int16) :: values(256)
        character(size(values)) :: str

        open(newunit = fileunit, file = filename, action = 'read', form = 'formatted')

        read(fileunit, *, end = 1) tokens_count
        allocate(tokenizer%tokens(tokens_count))

        token_index = 0
        do
            read(fileunit, *, end = 1) length
            read(fileunit, *, end = 1) values(:length)
            token_index = token_index + 1

            call int2char(values(:length), str)
            allocate(tokenizer%tokens(token_index)%value, source = str(:length))
        end do
        1 close(fileunit)

        tokenizer%root%char = char(0)
        tokenizer%root%index = 0
        do token_index = 1, size(tokenizer%tokens)
            call tokenizer%root%add(&
                    tokenizer%tokens(token_index)%value, 1, token_index)
        end do
    end function load_trie_tokenizer

    pure subroutine int2char(values, chars)
        integer(int16), intent(in) :: values(:)
        character(*), intent(out) :: chars
        integer :: i

        do concurrent (i = 1:size(values))
            chars(i:i) = char(values(i))
        end do
    end subroutine int2char

    subroutine trie_add(self, key, index, value_index)
        class(trie), intent(inout) :: self
        character(*), intent(in) :: key
        integer, intent(in) :: index
        integer, intent(in) :: value_index

        character :: c
        integer :: ic
        type(trie), pointer :: next => null()

        c = key(index:index)
        ic = ichar(c) + 1
        if (index > len(key)) then
            self%index = value_index
            return
        end if

        if (associated(self%next(ic)%p)) then
            next => self%next(ic)%p
        else
            allocate(next)
            next%char = c
            next%index = 0
            self%next(ic)%p => next
        end if

        call next%add(key, index + 1, value_index)
    end subroutine trie_add

    subroutine trie_print(self, padding)
        class(trie), intent(in) :: self
        integer, intent(in), optional :: padding
        integer :: spaces
        integer :: i, ic

        ic = ichar(self%char)

        spaces = 0
        if (present(padding)) spaces = padding
        write (*, '(a, z2.2)', advance = 'no') repeat(' ', spaces), ic

        if (ic > 32 .and. ic < 127) write (*, '(2x, a)', advance = 'no') self%char
        if (self%index > 0) then
            write (*, '(2x, i5)') self%index
        else
            write (*, *)
        end if

        do i = 1, size(self%next)
            if (associated(self%next(i)%p)) call self%next(i)%p%print(spaces + 2)
        end do
    end subroutine trie_print

    function trie_tokenizer_encode(self, text) result(tokens)
        class(trie_tokenizer), intent(in) :: self
        character(*), intent(in) :: text
        integer, allocatable :: tokens(:)
        integer :: text_index, token_index

        allocate(tokens(0))

        text_index = 1
        do while (text_index <= len(text))
            call self%root%find_longest(text, text_index, token_index)
            tokens = [tokens, token_index]
        end do
    end function trie_tokenizer_encode

    subroutine trie_find_longest(self, text, text_index, token_index)
        class(trie), target, intent(in) :: self
        character(*), intent(in) :: text
        integer, intent(inout) :: text_index
        integer, intent(out) :: token_index

        type(trie), pointer :: t
        integer :: i, ic

        token_index = 0
        t => self
        i = text_index
        ic = ichar(text(i:i)) + 1

        do while (associated(t%next(ic)%p))
            t => t%next(ic)%p
            i = i + 1

            if (t%index /= 0) then
                token_index = t%index
                text_index = i
            end if

            if (i > len(text)) return
            ic = ichar(text(i:i)) + 1
        end do
    end subroutine trie_find_longest

    pure function trie_tokenizer_decode(self, tokens) result(text)
        class(trie_tokenizer), intent(in) :: self
        integer, intent(in) :: tokens(:)
        character(:), allocatable :: text
        integer :: i, j, l

        l = 0
        do i = 1, size(tokens)
            l = l + len(self%tokens(tokens(i))%value)
        end do

        allocate(character(l) :: text)
        j = 1
        do i = 1, size(tokens)
            l = len(self%tokens(tokens(i))%value)
            text(j:j + l - 1) = self%tokens(tokens(i))%value
            j = j + l
        end do
    end function trie_tokenizer_decode

end module mod_trie_tokenizer