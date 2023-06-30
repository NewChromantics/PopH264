#pragma once

// a C++17 implementation of <span>
//	https://codereview.stackexchange.com/questions/217814/c17-span-implementation
#if __cplusplus >= 202002L
#include <span>
#define SPAN_HPP
#else

#include <array>       // for std::array, etc.
#include <cassert>     // for assert
#include <cstddef>     // for std::size_t, etc.
#include <iterator>    // for std::reverse_iterator, etc.
#include <type_traits> // for std::enable_if, etc.

#define CONSTRAINT(...) \
  std::enable_if_t<(__VA_ARGS__), int> = 0
#define EXPECTS(...) \
  assert((__VA_ARGS__))

namespace std {

  // constants

  // equivalent to std::numeric_limits<std::size_t>::max()
  inline constexpr std::size_t dynamic_extent = -1;

  // class template span

  template <class T, std::size_t N = dynamic_extent>
  class span;

  namespace span_detail {

    // detect specializations of span

    template <class T>
    struct is_span :std::false_type {};

    template <class T, std::size_t N>
    struct is_span<span<T, N>> :std::true_type {};

    template <class T>
    inline constexpr bool is_span_v = is_span<T>::value;

    // detect specializations of std::array

    template <class T>
    struct is_array :std::false_type {};

    template <class T, std::size_t N>
    struct is_array<std::array<T, N>> :std::true_type {};

    template <class T>
    inline constexpr bool is_array_v = is_array<T>::value;

    // ADL-aware data() and size()

    using std::data;
    using std::size;

    template <class C>
    constexpr decltype(auto) my_data(C& c)
    {
      return data(c);
    }

    template <class C>
    constexpr decltype(auto) my_size(C& c)
    {
      return size(c);
    }

    // detect container

    template <class C, class = void>
    struct is_cont :std::false_type {};

    template <class C>
    struct is_cont<C,
      std::void_t<
    std::enable_if_t<!is_span_v<C>>,
    std::enable_if_t<!is_array_v<C>>,
    std::enable_if_t<!std::is_array_v<C>>,
    decltype(data(std::declval<C>())),
        decltype(size(std::declval<C>()))
    >> :std::true_type {};

    template <class C>
    inline constexpr bool is_cont_v = is_cont<C>::value;
  }

  template <class T, std::size_t N>
  class span {
  public:    
    // constants and types

    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using index_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static constexpr index_type extent = N;

    // constructors, copy, and assignment

    // LWG 3198 applied
    constexpr span() noexcept
      : size_{0}, data_{nullptr}
    {
      static_assert(N == dynamic_extent || N == 0);
    }

    constexpr span(T* ptr, index_type n)
      : size_{n}, data_{ptr}
    {
      EXPECTS(N == dynamic_extent || N == n);
    }

    constexpr span(T* first, T* last)
      : size_{last - first}, data_{first}
    {
      EXPECTS(N == dynamic_extent || last - first = N);
    }

    template <std::size_t M,
          CONSTRAINT(N == dynamic_extent || N == M
             && std::is_convertible_v<std::remove_pointer_t<decltype(span_detail::my_data(std::declval<T(&)[M]>()))>(*)[], T(*)[]>)>
    constexpr span(T (&arr)[M]) noexcept
      : size_{M}, data_{arr}
    {
    }

    template <std::size_t M,
              CONSTRAINT(N == dynamic_extent || N == M
                 && std::is_convertible_v<std::remove_pointer_t<decltype(span_detail::my_data(std::declval<T(&)[M]>()))>(*)[], T(*)[]>)>
    constexpr span(std::array<value_type, M>& arr) noexcept
      : size_{M}, data_{arr.data()}
    {
    }

    template <std::size_t M,
              CONSTRAINT(N == dynamic_extent || N == M
                 && std::is_convertible_v<std::remove_pointer_t<decltype(span_detail::my_data(std::declval<T(&)[M]>()))>(*)[], T(*)[]>)>
    constexpr span(const std::array<value_type, M>& arr) noexcept
      : size_{M}, data_{arr.data()}
    {
    }

    template <class Cont,
          CONSTRAINT(N == dynamic_extent
             && span_detail::is_cont_v<Cont>
             && std::is_convertible_v<std::remove_pointer_t<decltype(span_detail::my_data(std::declval<Cont>()))>(*)[], T(*)[]>)>
    constexpr span(Cont& c)
      : size_{span_detail::my_size(c)},
        data_{span_detail::my_data(c)}
    {
    }

    template <class Cont,
          CONSTRAINT(N == dynamic_extent
             && span_detail::is_cont_v<Cont>
             && std::is_convertible_v<std::remove_pointer_t<decltype(span_detail::my_data(std::declval<Cont>()))>(*)[], T(*)[]>)>
    constexpr span(const Cont& c)
      : size_{span_detail::my_size(c)},
        data_{span_detail::my_data(c)}
    {
    }

    constexpr span(const span& other) noexcept = default;

    template <class U, std::size_t M,
          CONSTRAINT(N == dynamic_extent || N == M
             && std::is_convertible_v<U(*)[], T(*)[]>)>
    constexpr span(const span<U, M>& s) noexcept
      : size_{s.size()}, data_{s.data()}
    {      
    }

    ~span() noexcept = default;

    constexpr span& operator=(const span& other) noexcept = default;

    // subviews

    template <std::size_t Cnt>
    constexpr span<T, Cnt> first() const
    {
      EXPECTS(Cnt <= size());
      return {data(), Cnt};
    }

    template <std::size_t Cnt>
    constexpr span<T, Cnt> last() const
    {
      EXPECTS(Cnt <= size());
      return {data() + (size() - Cnt), Cnt};
    }

    template <std::size_t Off, std::size_t Cnt = dynamic_extent>
    constexpr auto subspan() const
    {
      EXPECTS(Off <= size() && (Cnt == dynamic_extent || Off + Cnt <= size()));
      if constexpr (Cnt != dynamic_extent)
        return span<T, Cnt>{data() + Off, Cnt};
      else if constexpr (N != dynamic_extent)
        return span<T, N - Off>{data() + Off, size() - Off};
      else
    return span<T, dynamic_extent>{data() + Off, size() - Off};
    }

    constexpr span<T, dynamic_extent> first(index_type cnt) const
    {
      EXPECTS(cnt <= size());
      return {data(), cnt};
    }

    constexpr span<T, dynamic_extent> last(index_type cnt) const
    {
      EXPECTS(cnt <= size());
      return {data() + (size() - cnt), cnt};
    }

    constexpr span<T, dynamic_extent> subspan(index_type off,
                          index_type cnt = dynamic_extent) const
    {
      EXPECTS(off <= size() && (cnt == dynamic_extent || off + cnt <= size()));
      return {data() + off, cnt == dynamic_extent ? size() - off : cnt};
    }

    // observers

    constexpr index_type size() const noexcept
    {
      return size_;
    }

    constexpr index_type size_bytes() const noexcept
    {
      return size() * sizeof(T);
    }

    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return size() == 0;
    }

    // element access

    constexpr reference operator[](index_type idx) const
    {
      EXPECTS(idx < size());
      return *(data() + idx);
    }

    constexpr reference front() const
    {
      EXPECTS(!empty());
      return *data();
    }

    constexpr reference back() const
    {
      EXPECTS(!empty());
      return *(data() + (size() - 1));
    }

    constexpr pointer data() const noexcept
    {
      return data_;
    }

    // iterator support

    constexpr iterator begin() const noexcept
    {
      return data();
    }

    constexpr iterator end() const noexcept
    {
      return data() + size();
    }

    constexpr const_iterator cbegin() const noexcept
    {
      return data();
    }

    constexpr const_iterator cend() const noexcept
    {
      return data() + size();
    }

    constexpr reverse_iterator rbegin() const noexcept
    {
      return reverse_iterator{end()};
    }

    constexpr reverse_iterator rend() const noexcept
    {
      return reverse_iterator{begin()};
    }

    constexpr const_reverse_iterator crbegin() const noexcept
    {
      return reverse_iterator{cend()};
    }

    constexpr const_reverse_iterator crend() const noexcept
    {
      return reverse_iterator{cbegin()};
    }

    friend constexpr iterator begin(span s) noexcept
    {
      return s.begin();
    }

    friend constexpr iterator end(span s) noexcept
    {
      return s.end();
    }

  private:
    pointer data_;
    index_type size_;
  };

  // deduction guide

  template <class T, std::size_t N>
  span(T (&)[N]) -> span<T, N>;

  template <class T, std::size_t N>
  span(std::array<T, N>&) -> span<T, N>;

  template <class T, std::size_t N>
  span(const std::array<T, N>&) -> span<const T, N>;

  template <class Cont>
  span(Cont&) -> span<typename Cont::value_type>;

  template <class Cont>
  span(const Cont&) -> span<const typename Cont::value_type>;

  // views of objects representation

  template <class T, std::size_t N>
  auto as_bytes(span<T, N> s) noexcept
    -> span<const std::byte,
        N == dynamic_extent ? dynamic_extent : sizeof(T) * N>
  {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size_bytes()};
  }

  template <class T, std::size_t N,
        CONSTRAINT(!std::is_const_v<T>)>
  auto as_writable_bytes(span<T, N> s) noexcept
    -> span<std::byte,
        N == dynamic_extent ? dynamic_extent : sizeof(T) * N>
  {
    return {reinterpret_cast<std::byte*>(s.data()), s.size_bytes()};
  }

}

namespace std {

  // tuple interface
  // the primary template declarations are included in <array>

  template <class T, std::size_t N>
  struct tuple_size<std::span<T, N>>
    : std::integral_constant<std::size_t, N> {};

  // not defined
  template <class T>
  struct tuple_size<std::span<T, std::dynamic_extent>>;

  template <std::size_t I, class T, std::size_t N>
  struct tuple_element<I, std::span<T, N>> {
    static_assert(N != std::dynamic_extent && I < N);
    using type = T;
  };

  template <std::size_t I, class T, std::size_t N>
  constexpr T& get(std::span<T, N> s) noexcept
  {
    static_assert(N != std::dynamic_extent && I < N);
    return s[I];
  }

}

#undef CONSTRAINT
#undef EXPECTS

#endif
