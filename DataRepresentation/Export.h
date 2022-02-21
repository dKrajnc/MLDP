/*!
 * \file
 * Storage-class information setting for DataRepresentation module.
 *
 * \remarks
 *
 * \authors
 * lpapp
 */

#pragma once

#if defined( EXPORT_DataRepresentation )
#	define DataRepresentation_API __declspec(dllexport)
#else
#	define DataRepresentation_API __declspec(dllimport)
#endif
