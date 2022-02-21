/*!
 * \file
 * Integral types for DataRepresentation module.
 *
 * \remarks
 *
 * \authors
 * lpapp
 */

#pragma once

#include <cstdint>

//-----------------------------------------------------------------------------

typedef unsigned int      uint;

// Integral types that match the word length.
#if defined( CC_MSVC )
typedef __int64           lint;
typedef unsigned __int64  ulint;
#else
typedef long int           lint;
typedef unsigned long int  ulint;
#endif
