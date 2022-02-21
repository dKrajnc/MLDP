#pragma once

#include <DataRepresentation/Export.h>
#include <DataRepresentation/Types.h>

namespace lpmldata
{

//-----------------------------------------------------------------------------

template < typename Type >
class DataRepresentation_API Array2D
{

public:

	Array2D( unsigned int aRowCount, unsigned int aColumnCount );

	virtual ~Array2D();

	const Type& operator () ( unsigned int aX, unsigned int aY ) const;

	Type& operator () ( unsigned int aX, unsigned int aY );

	void addEntry( unsigned int aX, unsigned int aY ) { operator() ( aX, aY ) += 1; }

	const unsigned int rowCount() const { return mRowCount; }

	const unsigned int columnCount() const { return mColumnCount; }

	const Type& at( unsigned int aX, unsigned int aY ) const { return operator()( aX, aY ); }

private:

	// to prevent unwanted copying:
	Array2D( const Array2D< Type >& );
	Array2D& operator = ( const Array2D< Type >& );

private:

	unsigned int mRowCount;
	unsigned int mColumnCount;
	Type* mArray;
};

//-----------------------------------------------------------------------------

}
