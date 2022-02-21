/*!
 * \file
 * Storage-class information setting for Evaluation module.
 *
 * \remarks
 *
 * \authors
 * lpapp
 */

#pragma once

#if defined( EXPORT_Evaluation )
#	define Evaluation_API __declspec(dllexport)
#else
#	define Evaluation_API __declspec(dllimport)
#endif
