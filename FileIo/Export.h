/*!
* \file
* Storage-class information setting for FileIo module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#if defined( EXPORT_FileIo )
#	define FileIo_API __declspec(dllexport)
#else
#	define FileIo_API __declspec(dllimport)
#endif
