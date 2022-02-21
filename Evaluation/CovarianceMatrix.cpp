#include <Evaluation/CovarianceMatrix.h>
#include <QSet>
#include <QDebug>
#include <omp.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

CovarianceMatrix::CovarianceMatrix( lpmldata::TabularData& aTabularData )
:
    lpmldata::Array2D< double >( aTabularData.columnCount(), aTabularData.columnCount() )
{
	initialize( aTabularData );
}

//-----------------------------------------------------------------------------

CovarianceMatrix::CovarianceMatrix( lpmldata::TabularData& aFirst, lpmldata::TabularData& aSecond )
:
	lpmldata::Array2D< double >( aFirst.columnCount(), aSecond.columnCount() )
{
	initialize( aFirst, aSecond );
}

//-----------------------------------------------------------------------------

CovarianceMatrix::~CovarianceMatrix()
{
}

//-----------------------------------------------------------------------------

void CovarianceMatrix::initialize( lpmldata::TabularData& aTabularData )
{
	QVector< double > means;
	int meansSize = aTabularData.columnCount();
	means.resize( meansSize );
	means.fill( 0.0 );

	// Go through all columns in the tabular data and determine means.
	for ( int columnIndex = 0; columnIndex < aTabularData.columnCount(); ++columnIndex )
	{
		means[ columnIndex ] = aTabularData.mean( columnIndex );
	}

	// Fill in the covariance matrix.
	omp_set_nested( 0 );

	for ( int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
	{
		QVariantList firstColumn = aTabularData.column( rowIndex );
#pragma omp parallel for schedule( guided )
		for ( int columnIndex = 0; columnIndex < columnCount(); ++columnIndex )
		{			
			QVariantList secondColumn = aTabularData.column( columnIndex );

			double leftSqrSum = 0.0;
			double rightSqrSum = 0.0;
			double numerator = 0.0;
			double denomiator = 0.0;
			double coefficient = 0.0;

			// Go through both column variables
			for ( int valueIndex = 0; valueIndex < firstColumn.size(); ++valueIndex )
			{
				double x = firstColumn.at( valueIndex ).toDouble();
				double y = secondColumn.at( valueIndex ).toDouble();

				double left  = x - means.at( rowIndex );
				double right = y - means.at( columnIndex );

				numerator   += left * right;
				leftSqrSum  += left * left;
				rightSqrSum += right * right;
			}

			denomiator  = sqrt( leftSqrSum * rightSqrSum );
			coefficient = numerator / denomiator;

#pragma omp critical
			{
				this->operator()( rowIndex, columnIndex ) = coefficient;				
			}
		}
	}
}

//-----------------------------------------------------------------------------


void CovarianceMatrix::initialize( lpmldata::TabularData& aFirst, lpmldata::TabularData& aSecond )
{
	QList< double > firstMeans;
	QList< double > secondMeans;

	// Go through all columns in the first tabular datas and determine means.
	for ( int columnIndex = 0; columnIndex < aFirst.columnCount(); ++columnIndex )
	{
		firstMeans.push_back( aFirst.mean( columnIndex ) );
	}

	// Go through all columns in the second tabular datas and determine means.
	for ( int columnIndex = 0; columnIndex < aSecond.columnCount(); ++columnIndex )
	{
		secondMeans.push_back( aSecond.mean( columnIndex ) );
	}

	// Fill in the covariance matrix.
	for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
	{
		for ( unsigned int columnIndex = 0; columnIndex < columnCount(); ++columnIndex )
		{
			double leftSqrSum = 0.0;
			double rightSqrSum = 0.0;
			double numerator = 0.0;
			double denomiator = 0.0;
			double coefficient = 0.0;

			QStringList firstKeys = aFirst.keys();
			QStringList secondKeys = aSecond.keys();
			QSet< QString > intersection = firstKeys.toSet().intersect( secondKeys.toSet() );

			QStringList intersectionOfKeys = intersection.toList();

			// Go through both column variables. 
			for ( int valueIndex = 0; valueIndex < intersectionOfKeys.size(); ++valueIndex )
			{
				
				double x = aFirst.valueAt(  intersectionOfKeys.at( valueIndex ), rowIndex    ).toString().replace( ",", "." ).toDouble();
				double y = aSecond.valueAt( intersectionOfKeys.at( valueIndex ), columnIndex ).toString().replace( ",", "." ).toDouble();

				if ( std::isnan( x ) || std::isnan( y ) ) continue;

				double left =  x - firstMeans.at( rowIndex );
				double right = y - secondMeans.at( columnIndex );

				numerator += left * right;

				leftSqrSum += left * left;
				rightSqrSum += right * right;

			}

			denomiator = sqrt( leftSqrSum * rightSqrSum );

			if ( denomiator == 0.0 )
			{
				coefficient = 0.0;
			}
			else
			{
				coefficient = numerator / denomiator;
			}

			this->operator()( rowIndex, columnIndex ) = coefficient;
		}
	}
}

//-----------------------------------------------------------------------------

}