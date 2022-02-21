#include <Evaluation/PCA.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

void PCA::build( const lpmldata::DataPackage& aDataPackage )
{
	//Create a matrix from original dataset
	auto FDB     = aDataPackage.featureDatabase();
	auto headers = FDB.headerNames();
	auto keys    = aDataPackage.commonKeys();		

	std::sort( keys.begin(), keys.end() );

	auto rowsCount    = keys.size();
	auto columnsCount = headers.size();

	lpmldata::Array2D< double > originalMatrix( rowsCount, columnsCount );		

	for ( int i = 0; i < rowsCount; ++i ) 
	{
		for ( int j = 0; j < columnsCount; ++j )
		{
			auto key = keys.at( i );
			auto cellValue = FDB.valueAt( key, j ).toDouble();
			
			originalMatrix.operator()( i, j ) = cellValue;		
		}				
	}

	//Recenter the dataset
	auto recenteredFDB = recenter( aDataPackage );
	

	//Compute eigenvalues and eigenvectors
	auto eigenParams = eigens( recenteredFDB );
	auto eigenValues = eigenParams.keys();
		
		
	//Calculate the sum of total variance
	double sum = 0;

	for ( int i = 0; i < eigenValues.size(); ++i )
	{		
		sum += std::abs( eigenValues.at( i ) );
	}

	//Compute the principal component variance percentage
	QMap< double, double > variances;
	
	for ( auto& element : eigenValues )
	{
		auto value = std::abs( ( element / sum ) * 100 );

		variances.insertMulti( element, value );
	}
	
	//Determine desired principal components	
	QList< double > choosenEigenValues;

	QMapIterator< double, double > itr( variances );

	double total = 0;

	itr.toBack();
	while ( itr.hasPrevious() )
	{
		itr.previous();

		auto value = itr.value();
		auto key   = itr.key();
		
		if ( total <= mPreservationPercentage )
		{
			choosenEigenValues.push_back( key );

			total += value;
		}
		else
		{
			break;
		}
	}

	//Determine desired eigen components for new dataset transformation

	for ( int j = 0; j < choosenEigenValues.size(); ++j )
	{
		auto choosen = choosenEigenValues.at( j );

		for ( int i = 0; i < eigenParams.size(); ++i )
		{			
			auto overall = eigenParams.keys().at( i );
			auto vector  = eigenParams.values().at( i );

			if ( overall == choosen )
			{
				mChoosenEigenvectors.push_back( vector );
			}
		}
	}
	
}

//-----------------------------------------------------------------------------

lpmldata::TabularData PCA::recenter( const lpmldata::DataPackage& aDataPackage )
{
	
	auto FDB        = aDataPackage.featureDatabase();
	auto headers    = FDB.headerNames();

	//Read out the feature values for each sample from the dataset 
	QVector< QVector< double > > modifiedFeatureValues;
	QVector< QString > keys;
	
	for ( int i = 0; i < 1; ++i )
	{
		for ( int j = 0; j < FDB.keys().size(); ++j )
		{
			auto key =  FDB.keys().at( j );

			keys.push_back( key );
		}
	}

	std::sort( keys.begin(), keys.end() );

	for ( int i = 0; i < headers.size(); ++i )
	{
		QVector< double > featureValues;
		
		for ( int j = 0; j < FDB.keys().size(); ++j )
		{
			auto key          = keys.at( j );
			auto featureValue = FDB.valueAt( key, i ).toDouble();
			
			featureValues.push_back( featureValue );
		}		
		

		//Calculate mean value for each feature	
		auto denominator        = featureValues.size();	

		auto sumOfFeatureValues = std::accumulate( featureValues.begin(), featureValues.end(), 0.0 );

		auto mean               = sumOfFeatureValues / denominator;		


		//Substract mean value from the feature values
		QVector< double >substractedFeatureValues;

		for ( double& element : featureValues )
		{
			substractedFeatureValues.push_back( element - mean ); 
		}

		modifiedFeatureValues.push_back( substractedFeatureValues );
	}
	

	//Return the recentered dataset
	lpmldata::TabularData updatedFDB = aDataPackage.featureDatabase();
	return updatedFDB;
}

//-----------------------------------------------------------------------------

QMap< double, QVector< double > > PCA::eigens( lpmldata::TabularData& aDataBase )
{
	QMap< double, QVector< double > > eigen;
	QVector< double > eigenValues;
	QVector< QVector< double > > eigenVectors;


	//Compute a covariance matrix
	lpmleval::CovarianceMatrix covarianceMatrix( aDataBase );
	

	//The QR algorithm
	/*The QR algorithm requires the orthonormal matrix Q and upper triangular matrix R, in order to be able to calculate the eigenvalues and eigenvectors of the original matrix A. 
	  In order to obtain the Q matrix, the Gram-Smidt process is applied.
	  After matrices are acquired, we multiply them iteratively, such as: A = Q * R => A' = R * Q' => A" = R' * Q", etc. */

	//Create a copy of covariance matrix and use it as initial matrix
	lpmldata::Array2D< double > newMatrix( covarianceMatrix.rowCount(), covarianceMatrix.columnCount() );
	
	for ( int i = 0; i < covarianceMatrix.rowCount(); ++i )
	{
		for ( int j = 0; j < covarianceMatrix.columnCount(); ++j )
		{
			auto cellValue = covarianceMatrix.at( i, j );

			newMatrix.operator()( i, j ) = cellValue;
		}
	}

	lpmldata::Array2D< double > q_Matrix( newMatrix.rowCount(), newMatrix.columnCount() );

	int iterationNumber = 10; // The QR algorithm iteration number. Parameter?   

	for ( int index = 0; index < iterationNumber; ++index ) 
	{		
		QVector< QVector< double > > initialVectors; // original matrix
		QVector< QVector< double > > orthonormalVectors; // Q matrix

		//Create the initial orthonormal vector which shall represent the first column in the orthonormal matrix (Q) using Gram-Smidt process 
		for ( int i = 0; i < 1; ++i ) //The first orthonormal vector of the Q matrix is equal to the first vector of original A matrix
		{
			QVector< double > orthonormal;

			for ( int j = 0; j < newMatrix.rowCount(); ++j )
			{

				auto cellValue = newMatrix.at( i, j );

				orthonormal.push_back( cellValue );
			}

			initialVectors.push_back( orthonormal );

			auto normalizedVector = normalize( orthonormal ); //should it be normalized just in first iteration e.g. if( index == 0 )? does this count in the validation data processing
															  
			orthonormalVectors.push_back( normalizedVector );


		}


		//Create the set of orthonormal vectors after the initial column which shall represent the rest of orthonormal matrix (Q) using Gram-Smidt process 
		for ( int i = 1; i < newMatrix.columnCount(); ++i )
		{
			QVector< double > orthonormal;
			QVector< double > initial;
			QVector< QVector< double > > projectionValues;

			for ( int j = 0; j < newMatrix.rowCount(); ++j )
			{

				auto cellValue = newMatrix.at( i, j );

				initial.push_back( cellValue );
			}

			initialVectors.push_back( initial );

			for ( int index = 0; index < i; ++index )
			{
				auto projectionValue = projection( initial, orthonormalVectors.at( index ) );

				projectionValues.push_back( projectionValue );
			}
					   
			for ( int j = 0; j < projectionValues.size(); ++j )
			{
				auto subtractors   = projectionValues.at( j );
				double substracted = 0;
				auto size          = projectionValues.size();

				for ( int m = 0; m < initial.size(); m++ )
				{
					substracted = initial.at( m ) - subtractors.at( m );

					initial.replace( m, substracted );

					if ( j == size - 1 )
					{
						orthonormal.push_back( substracted );
					}

				}
			}

			auto normalizedVector = normalize( orthonormal );//should it be normalized just in first iteration e.g. if( index == 0 )?does this count in the validation data processing
															  
			orthonormalVectors.push_back( normalizedVector );
		}
			

		//Create upper triangular matrix (R)
		auto rowsNumber    = orthonormalVectors.at( 0 ).size(); // rowsCount
		auto columnsNumber = orthonormalVectors.size(); //columnsCount

		lpmldata::Array2D< double > upperTriangularMatrix( rowsNumber, columnsNumber );


		//Filling upper triangular matrix (R)
		for ( int i = 0; i < rowsNumber; ++i )
		{
			for ( int j = 0; j < columnsNumber; ++j )
			{
				auto orthonormal = orthonormalVectors.at( i );
				auto initial     = initialVectors.at( j );
				double sum       = 0;

				for ( int k = 0; k < orthonormal.size(); ++k )
				{
					auto value = orthonormal.at( k ) * initial.at( k );

					sum += value;
				}

				if ( i > j )
				{
					upperTriangularMatrix.operator()( i, j ) = 0; //rounding extremely small values to 0
				}
				else
				{
					upperTriangularMatrix.operator()( i, j ) = sum;
				}

			}
		}


		//Create new matrix (A)	- newly obtained initial matrix 
		for ( int i = 0; i < upperTriangularMatrix.rowCount(); ++i )
		{
			auto orthonormal = orthonormalVectors.at( i );

			for ( int j = 0; j < upperTriangularMatrix.columnCount(); ++j )
			{
				double sum = 0;

				for ( int k = 0; k < upperTriangularMatrix.columnCount(); ++k )
				{
					auto orthonormalValue = orthonormal.at( k );
					auto triangularValue  = upperTriangularMatrix.at( j, k );

					auto value = triangularValue * orthonormalValue; //Multiplying R' and Q', read the QR algorithm iteration process

					sum += value;
				}

				newMatrix.operator()( i, j ) = sum;

			}
		}


		//Get eigenvectors
		if ( index == 0 )
		{
			for ( int i = 0; i < columnsNumber; ++i )
			{
				auto vector = orthonormalVectors.at( i );

				for ( int j = 0; j < rowsNumber; ++j )
				{
					auto element = vector.at( j );

					q_Matrix.operator()( i, j) = element;
				}
			}			
		}
		else					
		{
			for ( int i = 0; i < q_Matrix.rowCount(); ++i )
			{				
				auto vector = orthonormalVectors.at( i );

				for ( int j = 0; j < q_Matrix.columnCount(); ++j )
				{					
					double sum = 0;

					for ( int k = 0; k < q_Matrix.columnCount(); ++k )
					{
						auto matrixElement = q_Matrix.at( k, j );
						auto vectorElement = vector.at( k );

						auto value = matrixElement * vectorElement;

						sum += value;
					}					
					q_Matrix.operator()( i, j ) = sum;
				}			
			}
		}

		for ( int i = 0; i < q_Matrix.columnCount(); ++i )
		{
			QVector< double > temporary;
			
			for ( int j = 0; j < q_Matrix.rowCount(); ++j )
			{
				auto value = q_Matrix.at( i, j );

				temporary.push_back( value );
			}

			if ( index == iterationNumber - 1 )
			{
				eigenVectors.push_back( temporary );				
			}
		}
	}


	//Extract eigenvalues
	for ( int i = 0; i < newMatrix.columnCount(); ++i )
	{	
		for ( int j = 0; j < newMatrix.rowCount(); ++j )
		{
			if ( i == j )
			{				
				eigenValues.push_back( newMatrix.at( i, j ) );				
			}			
		}
	}


	//Store eigenvalue-eigenvector pair 
	for ( int i = 0; i < eigenVectors.size(); ++i )
	{
		auto eigenVec = eigenVectors.at( i );
		auto eigenVal = eigenValues.at( i );

		eigen.insertMulti( eigenVal, eigenVec );
	}

	return eigen;
}

//-----------------------------------------------------------------------------

QVector< double > PCA::projection( QVector< double >& aInitial, const QVector< double >& aOrthonormal )
{
	QVector< double > projectionValues;
	
	double sumNumerator   = 0;
	double sumDenominator = 0;

	for ( int i = 0; i < aInitial.size(); i++ )
	{
		auto numerator   = aInitial.at( i ) * aOrthonormal.at( i );		
		auto denominator = pow( aOrthonormal.at( i ), 2 );

		sumNumerator   += numerator;
		sumDenominator += denominator;
	}

	auto devidedValue = sumNumerator / sumDenominator;

	for ( int i = 0; i < aInitial.size(); i++ )
	{
		auto result = devidedValue * aOrthonormal.at( i );
		
		projectionValues.push_back( result );
	}

	return projectionValues;
}

//-----------------------------------------------------------------------------

QVector< double > PCA::normalize( QVector< double >& aOrthonormal )
{
	QVector< double > normalizedValues;

	double squaredValue = 0;

	for ( int i = 0; i < aOrthonormal.size(); ++i )
	{
		auto value = pow( aOrthonormal.at( i ), 2 );

		squaredValue += value;
	}

	auto root = sqrt( squaredValue );

	for ( auto& element : aOrthonormal )
	{
		normalizedValues.push_back( element / root );
	}

	return normalizedValues;
}
	
//-----------------------------------------------------------------------------

QVector< QVector< double > > PCA::featureNormalization( QVector< QVector< double > >& aFeatureVector )
{
	QList< double > means;
	QList< double > deviations;
	QVector< QVector< double > > normalizedFeatures;

	for ( int i = 0; i < aFeatureVector.size(); ++i )
	{
		auto meanValue = mean( aFeatureVector.at( i ) );
		auto deviation = standardDeviation( aFeatureVector.at( i ), meanValue );

		means.push_back( meanValue );
		deviations.push_back( deviation );		
	}	

	for ( int i = 0; i < aFeatureVector.size(); ++i )
	{
		QVector< double > normalizedVector;

		for ( int j = 0; j < aFeatureVector.at( i ).size(); ++j )
		{
			auto element = aFeatureVector.at( i ).at( j );

			auto subtracted = element    - means.at( i ); 
			auto devided    = subtracted / deviations.at( i );

			normalizedVector.push_back( devided );
		}
		normalizedFeatures.push_back( normalizedVector );
	}
	return normalizedFeatures;
}

//-----------------------------------------------------------------------------

double PCA::mean( const QVector< double >& aFeatureVector )
{
	double mean      = 0.0;
	double sum       = 0.0;
	auto denominator = aFeatureVector.size();
		
	std::for_each( aFeatureVector.begin(), aFeatureVector.end(),
		[ &sum ]( auto x ) { sum += x; } );

	mean = sum / denominator;			

	return mean;
}

//-----------------------------------------------------------------------------

double PCA::standardDeviation( const QVector< double >& aFeatureVector, double aMeanValue )
{
	//Calculate Variance
	QVector< double > values;
	double variance = 0.0;
	double standardDeviation = 0.0;

	for ( int i = 0; i < aFeatureVector.size(); ++i )
	{
		auto element = aFeatureVector.at( i );
		auto mean    = aMeanValue;

		auto substracted = element - mean;
		auto squared     = std::pow( substracted, 2 );

		values.push_back( squared );
	}

	std::for_each( values.begin(), values.end(),
		[ &variance ]( auto x ) { variance += x; } );

	//Calculate standard deviation
	standardDeviation = std::sqrt( variance );

	return standardDeviation;	
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage PCA::run( const lpmldata::DataPackage& aDataPackage )
{
	auto FDB = aDataPackage.featureDatabase();
	auto headers = FDB.headerNames();
	auto keys = aDataPackage.commonKeys();

	if ( mChoosenEigenvectors.isEmpty() )
	{
		for ( auto& feature : headers )
		{
			QVector< double > featureVector;

			for ( auto& key : keys )
			{				
				featureVector.push_back( FDB.valueAt( key, headers.indexOf( feature ) ).toDouble() );
			}
			mChoosenEigenvectors.push_back( featureVector );
		}
		lpmldata::DataPackage result( aDataPackage.featureDatabase(), aDataPackage.labelDatabase() );

		return result;
	}
	else
	{	

		lpmldata::Array2D< double > originalMatrix( keys.size(), headers.size() );

		for ( int i = 0; i < keys.size(); ++i )
		{
			for ( int j = 0; j < headers.size(); ++j )
			{
				auto key = keys.at( i );
				auto cellValue = FDB.valueAt( key, j ).toDouble();

				originalMatrix.operator()( i, j ) = cellValue;
			}
		}


		//Create new feature names
		QStringList featureNames;

		for ( int i = 0; i < mChoosenEigenvectors.size(); ++i )
		{
			QString name = "A::B::Feature" + QString::number( i + 1 );

			featureNames.append( name );
		}


		//Create new dataset
		QVector< QVector< double > > features;

		for ( int i = 0; i < mChoosenEigenvectors.size(); ++i )
		{
			auto vector = mChoosenEigenvectors.at( i );

			QVector< double > feature;

			for ( int j = 0; j < originalMatrix.rowCount(); ++j )
			{
				double sum = 0.0;

				for ( int k = 0; k < vector.size(); ++k )
				{
					auto originalValue = originalMatrix.at( j, k );
					auto element = vector.at( k );

					auto value = originalValue * element;

					sum += value;
				}
				feature.push_back( sum );
			}
			features.push_back( feature );
		}
				
		//Save the obtained dataset
		lpmldata::TabularData updatedFDB = aDataPackage.featureDatabaseSubset( features, featureNames );

		lpmldata::DataPackage result( updatedFDB, aDataPackage.labelDatabase() );

		return result;
	}


}

//-----------------------------------------------------------------------------

}