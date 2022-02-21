#include <DataRepresentation/Array2D.h>

namespace lpmldata
{

//-----------------------------------------------------------------------------
template < typename Type >
Array2D< Type >::Array2D( unsigned int aRowCount, unsigned int aColumnCount )
:
	mRowCount( aRowCount ),
	mColumnCount( aColumnCount )
{
	if ( aRowCount > 0 && aColumnCount > 0 ) mArray = new Type[ aRowCount * aColumnCount ];
	for ( unsigned int arrayIndex = 0; arrayIndex < aRowCount * aColumnCount; ++arrayIndex )
	{
		mArray[ arrayIndex ] = 0;
	}
}

//-----------------------------------------------------------------------------
template < typename Type >
Array2D< Type >::~Array2D()
{
	delete[] mArray;
}

//-----------------------------------------------------------------------------
template < typename Type >
const Type& Array2D< Type >::operator() ( unsigned int aX, unsigned int aY ) const
{
	return mArray[ aY*mRowCount + aX ];
}

//-----------------------------------------------------------------------------
template < typename Type >
Type& Array2D< Type >::operator() ( unsigned int aX, unsigned int aY )
{
	return mArray[ aY*mRowCount + aX ];
}

//-----------------------------------------------------------------------------

// Template instantiations to export:
template class DataRepresentation_API Array2D< unsigned int >;
template class DataRepresentation_API Array2D< float >;
template class DataRepresentation_API Array2D< double >;

}