#include <Evaluation/TabularDataFilter.h>
#include <Evaluation/FeatureSelector.h>
#include <QSet>
#include <omp.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

TabularDataFilter::TabularDataFilter()
:
	mRandomGenerator()
{
	std::random_device rd;
	mRandomGenerator = std::mt19937_64( rd() );
	mAbc = "abcdefghijklmnopqrstuvwxyz";
}

//-----------------------------------------------------------------------------

void TabularDataFilter::eraseIncompleteRecords( lpmldata::TabularData& aFeatureDatabase )
{
	auto keys = aFeatureDatabase.keys();
	QStringList keysToDelete;

	for ( auto key : keys )
	{
		auto featureVector = aFeatureDatabase.value( key );
		if ( featureVector.contains( "NA" ) || featureVector.contains( "nan" ) )
		{
			keysToDelete.push_back( key );
		}
	}

	for ( auto key : keysToDelete )
	{
		aFeatureDatabase.remove( key );
	}
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subtableByKeyExpression( lpmldata::TabularData& aFeatureDatabase, QString aKeyExpression )
{
	lpmldata::TabularData reducedData;
	reducedData.header() = aFeatureDatabase.header();

	QStringList keys = aFeatureDatabase.keys();

	for ( auto key : keys )
	{
		if ( key.contains( aKeyExpression ) )
		{
			QVariantList inputFeatureVector = aFeatureDatabase.value( key );
			reducedData.insert( key, inputFeatureVector );
		}
	}

	return reducedData;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subtableByReductionCovaraince( lpmldata::TabularData& aFeatureDatabase, lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, double aSimilarityThreshold )
{
	lpmldata::TabularData reducedData;

	FeatureSelector fs( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	QVector< double > mask;
	mask = fs.executeFrCovMxGlobal( aSimilarityThreshold );
	QStringList inputHeader = aFeatureDatabase.headerNames();
	QStringList outputHeader;

	// Go through headers and generate a new header based on mask.
	for ( int i = 0; i < mask.size(); ++i )
	{
		if ( mask.at( i ) > 0.0 )
		{
			outputHeader.push_back( inputHeader.at( i ) );
		}
	}

	reducedData.setHeader( outputHeader );

	// Go through the keys and build up a new feature vector based on mask.
	QStringList keys = aFeatureDatabase.keys();

	for ( auto key : keys )
	{
		QVariantList inputFeatureVector = aFeatureDatabase.value( key );
		QVariantList outputFeatureVector;
		for ( int i = 0; i < mask.size(); ++i )
		{
			if ( mask.at( i ) > 0.0 )
			{
				outputFeatureVector.push_back( inputFeatureVector.at( i ) );
			}
		}
		reducedData.insert( key, outputFeatureVector );
		//qDebug() << "Covariance reduction - Finished key " << key;
	}

	return reducedData;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subtableByReductionKDE( lpmldata::TabularData& aFeatureDatabase, lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aFeatureCount )
{
	lpmldata::TabularData reducedData;

	FeatureSelector fs( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	QVector< double > mask;
	mask = fs.executeFsKdeOverlap( aFeatureCount );
	QStringList inputHeader = aFeatureDatabase.headerNames();
	QStringList outputHeader;

	// Go through headers and generate a new header based on mask.
	for ( int i = 0; i < mask.size(); ++i )
	{
		if ( mask.at( i ) > 0.0 )
		{
			outputHeader.push_back( inputHeader.at( i ) );
		}
	}

	reducedData.setHeader( outputHeader );

	// Go through the keys and build up a new feature vector based on mask.
	QStringList keys = aFeatureDatabase.keys();

	for ( auto key : keys )
	{
		QVariantList inputFeatureVector = aFeatureDatabase.value( key );
		QVariantList outputFeatureVector;
		for ( int i = 0; i < mask.size(); ++i )
		{
			if ( mask.at( i ) > 0.0 )
			{
				outputFeatureVector.push_back( inputFeatureVector.at( i ) );
			}
		}
		reducedData.insert( key, outputFeatureVector );
		//qDebug() << "KDE reduction - Finished key " << key;
	}

	return reducedData;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::foldConfigTable( int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex )
{
	lpmldata::TabularData foldTable;
	QMap< int, QStringList > foldSelectorMap;

	// Generate the fold selector map.

	foldSelectorMap = this->stratifiedKFoldMap( aFoldCount, aSubsetRatio, aFeatureDatabase, aLabelDatabase, aLabelIndex );

	// Initiate the foldTable header.
	lpmldata::TabularDataHeader header;

	for ( int headerIndex = 0; headerIndex < aFoldCount; ++headerIndex )
	{
		QVariantList headerValue = { headerIndex, "Float" };
		header.insert( QString::number( headerIndex ), headerValue );
	}

	foldTable.header() = header;

	// Determine common keys from FDB and LDB based on labelindex.
	QStringList keys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );

	// Go through the keys and folds and figure out if the key in the given fold is included --> Value 1, otherwise 0.

	for ( auto &key : keys )
	{
		QVariantList foldRow;
		for ( int i = 0; i < aFoldCount; ++i )
		{
			if ( foldSelectorMap.value( i ).contains( key ) && aSubsetRatio > 0.0 )  // Exclude from given fold.
			{
				foldRow.push_back( 1 );
			}
			else
			{
				foldRow.push_back( 0 );
			}
		}

		foldTable.insert( key, foldRow );
	}

	return foldTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::foldConfigTableGenerator( int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, FoldSelector aFoldSelector )
{
	QStringList intersectKeys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	lpmldata::TabularData filteredLabelDatabase = this->subTableByKeys( aLabelDatabase, intersectKeys );
	QStringList labelGroups = this->labelGroups( filteredLabelDatabase, aLabelIndex );

	// Initialize the table
	lpmldata::TabularData foldConfigTable;
	QStringList headerNames;
	for ( int i = 0; i < aFoldCount; ++i )
	{
		headerNames.push_back( "Fold-" + QString::number( i + 1 ) );
	}
	foldConfigTable.setHeader( headerNames );

	// Fill up the table with empty values.
	for ( auto key : intersectKeys )
	{
		QVariantList vector;
		for ( int i = 0; i < aFoldCount; ++i )
		{
			vector.push_back( 0 );
		}
		foldConfigTable.insert( key, vector );
	}

	QMap< int, QStringList > labelGroupKeys;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		labelGroupKeys.insert( i, this->keysByLabelGroup( filteredLabelDatabase, aLabelIndex, labelGroups.at( i ) ) );
	}

	switch ( aFoldSelector )
	{
	case FoldSelector::LOO:
	{
		break;
	}
	case FoldSelector::KFold:
	{
		break;
	}
	case FoldSelector::MCStratified:
	{
		break;
	}
	case FoldSelector::MCEqual:
	{
		break;
	}
	}

	//Determine minority sample count, take at least 1 sample.
	int minorityGroupSize = filteredLabelDatabase.rowCount();  // This is higher than the maximum, unless we have only one label outcome.
	int minorityGroupIndex = -1;
	QVector< int > originalSamplecounts;

	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		originalSamplecounts.push_back( labelGroupKeys.value( i ).size() );

		if ( labelGroupKeys.size() < minorityGroupSize )
		{
			minorityGroupSize = labelGroupKeys.value( i ).size();
			minorityGroupIndex = i;
		}
	}

	int minorityGroupSizeValdiation = std::max( 1, int( minorityGroupSize * aSubsetRatio ) );  // This is needed to make sure that at least 1 element is chosen from the minority group for validation.

	// Determine all other sample counts relative to the minority sample counts.
	QVector< int > subsetSampleCountsValidation;
	QVector< int > subsetSampleCountsTraining;

	subsetSampleCountsValidation.resize( labelGroupKeys.size() );
	subsetSampleCountsTraining.resize( labelGroupKeys.size() );

	

	// Calculate the sample sizes for the training and validation sets. 
	switch ( aFoldSelector )
	{
	case FoldSelector::KFold:
	{
		double correctedSubsetRatio = double( minorityGroupSizeValdiation ) / double( minorityGroupSize );  // Corrected subset ratio based on corrected minority subset size.
		subsetSampleCountsValidation[ minorityGroupIndex ] = minorityGroupSizeValdiation;
		for ( int i = 0; i < labelGroups.size(); ++i )
		{
			if ( i != minorityGroupIndex )  // Check all groups except the minority one.
			{
				subsetSampleCountsValidation[ i ] = originalSamplecounts.at( i ) * correctedSubsetRatio;
			}
		}
		for ( int i = 0; i < labelGroups.size(); ++i )
		{
			subsetSampleCountsTraining[ i ] = originalSamplecounts.at( i ) - subsetSampleCountsValidation.at( i );
		}
		break;
	}
	case FoldSelector::MCEqual:
	{
		int trainingSize = minorityGroupSize - minorityGroupSizeValdiation;
		for ( int i = 0; i < labelGroups.size(); ++i )
		{
			subsetSampleCountsValidation[ i ] = minorityGroupSizeValdiation;
			subsetSampleCountsTraining[ i ]   = trainingSize;
		}
		break;
	}
	}

	


	// Generate the fold configuration data


	//TODO: Here different subsamplers shall be called (k-fold, equal subsample, LOO, etc).
	for ( int foldIndex = 0; foldIndex < aFoldCount; ++foldIndex )
	{
		for ( int i = 0; i < labelGroups.size(); ++i )
		{		
			std::shuffle( labelGroupKeys[ i ].begin(), labelGroupKeys[ i ].end(), mRandomGenerator );  // Shuffle the label group keys.
			QStringList keys = labelGroupKeys.value( i );

			int sampleIndex = 0;
			for ( auto key : keys )
			{		
				if ( sampleIndex < subsetSampleCountsValidation.at( i ) )  // Take out validation samples, value them with 1.
				{
					foldConfigTable.valueAt( key, foldIndex ) = 1;
				}
				else if ( sampleIndex < subsetSampleCountsValidation.at( i ) + subsetSampleCountsTraining.at( i ) )  // Take out training samples, value them with 0.
				{
					foldConfigTable.valueAt( key, foldIndex ) = 0;
				}
				else
				{
					foldConfigTable.valueAt( key, foldIndex ) = -1;  // All other samples with value -1
				}
				
				++sampleIndex;
			}		
		}
	}
	
	return foldConfigTable;

}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::leaveOneOutSubSampler( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex )
{
	QStringList intersectKeys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );

	// Initialize the table
	lpmldata::TabularData foldConfigTable;
	QStringList headerNames;
	for ( int i = 0; i < intersectKeys.size(); ++i )
	{
		headerNames.push_back( "Fold-" + QString::number( i + 1 ) );
	}
	foldConfigTable.setHeader( headerNames );

	// Fill up the table
	int keyIndex = 0;
	for ( auto key : intersectKeys )
	{                                    
		// Initiate the fold vector.
		QVariantList vector;
		for ( int i = 0; i < intersectKeys.size(); ++i )
		{
			vector.push_back( 0 );
		}

		// The given key is valdiation key.
		vector[ keyIndex ] = 1;
		foldConfigTable.insert( key, vector );
		++keyIndex;
	}

	return foldConfigTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::stratifiedKFoldSubSampler( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aFoldCount )
{
	QStringList intersectKeys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	lpmldata::TabularData filteredLabelDatabase = this->subTableByKeys( aLabelDatabase, intersectKeys );
	QStringList labelGroups = this->labelGroups( filteredLabelDatabase, aLabelIndex );

	// Initialize the table
	lpmldata::TabularData foldConfigTable;
	QStringList headerNames;
	for ( int i = 0; i < aFoldCount; ++i )
	{
		headerNames.push_back( "Fold-" + QString::number( i + 1 ) );
	}
	foldConfigTable.setHeader( headerNames );

	// Fill up the table with empty values.
	for ( auto key : intersectKeys )
	{
		QVariantList vector;
		for ( int i = 0; i < aFoldCount; ++i )
		{
			vector.push_back( 0 );
		}
		foldConfigTable.insert( key, vector );
	}

	// Build up a labelgroup key map.
	QMap< int, QStringList > labelGroupKeys;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		labelGroupKeys.insert( i, this->keysByLabelGroup( filteredLabelDatabase, aLabelIndex, labelGroups.at( i ) ) );
	}

	//Determine minority sample count, take at least 1 sample.
	int minorityGroupSize = filteredLabelDatabase.rowCount();  // This is higher than the maximum, unless we have only one label outcome.
	int minorityGroupIndex = -1;
	QVector< int > originalSamplecounts;

	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		originalSamplecounts.push_back( labelGroupKeys.value( i ).size() );

		if ( labelGroupKeys.value( i ).size() < minorityGroupSize )
		{
			minorityGroupSize = labelGroupKeys.value( i ).size();
			minorityGroupIndex = i;
		}
	}

	int minorityGroupSizeValdiation = std::max( 1, int( minorityGroupSize * ( 1.0 / double( aFoldCount ) ) ) );  // This is needed to make sure that at least 1 element is chosen from the minority group for validation.

	// Determine all other sample counts relative to the minority sample counts.
	QVector< int > subsetSampleCountsValidation;
	QVector< int > subsetSampleCountsTraining;

	subsetSampleCountsValidation.resize( labelGroupKeys.size() );
	subsetSampleCountsTraining.resize( labelGroupKeys.size() );

	// Calculate the sample sizes for the training and validation sets. 
	int trainingSize = minorityGroupSize - minorityGroupSizeValdiation;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		subsetSampleCountsValidation[ i ] = minorityGroupSizeValdiation;
		subsetSampleCountsTraining[ i ] = trainingSize;
	}

	//Fill up the foldconfig table.
	QVector< int > labelGroupIndices;
	labelGroupIndices.resize( labelGroupKeys.size() );
	labelGroupIndices.fill( 0 );

	for ( int foldIndex = 0; foldIndex < aFoldCount; ++foldIndex )
	{
		QMap< QString, int > currentFoldConfig;

		for ( int i = 0; i < labelGroups.size(); ++i )
		{
			QStringList keys = labelGroupKeys.value( i );

			// Determine the validation samples of the given subgroup.
			for ( int validationIndex = 0; validationIndex < subsetSampleCountsValidation.at( i ); ++validationIndex )
			{	
				currentFoldConfig.insert( keys.at( labelGroupIndices.at( i ) ), 1 );
				++labelGroupIndices[ i ];
				if ( labelGroupIndices.at( i ) == keys.size() )
				{
					labelGroupIndices[ i ] = 0;
				}
			}

			// Determine the training samples of the given subgroup.
			for ( auto key : keys )
			{
				if ( !currentFoldConfig.contains( key ) )
				{

					currentFoldConfig.insert( key, 0 );
				}
			}

			// Save the fold config to the table.
			for ( auto key : currentFoldConfig.keys() )
			{
				foldConfigTable.valueAt( key, foldIndex ) = currentFoldConfig.value( key );
			}
		}
	}

	// Check if there are samples that have never been part of validation.
	QStringList neverSelectedForValidation;
	for ( auto key : foldConfigTable.keys() )
	{
		bool isNeverSelectedForValdiation = true;
		QVariantList foldRow = foldConfigTable.value( key );
		for ( auto foldValue : foldRow )
		{
			if ( foldValue.toInt() == 1 )
			{
				isNeverSelectedForValdiation = false;
			}
		}

		if ( isNeverSelectedForValdiation )
		{
			neverSelectedForValidation.push_back( key );
		}
	}

	// Save the excluded keys for validation.
	int foldIndexForInsert = 0;
	for ( int i = 0; i < neverSelectedForValidation.size(); ++i )
	{
		QString validationKey = neverSelectedForValidation.at( i );
		foldConfigTable.valueAt( validationKey, foldIndexForInsert ) = 1;
		++foldIndexForInsert;
		if ( foldIndexForInsert == aFoldCount )
		{
			foldIndexForInsert = 0;
		}
	}

	return foldConfigTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::ballancedPermutationSubSampler( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aMaximumFoldCount, double aSubsetRatio )
{
	QStringList intersectKeys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	lpmldata::TabularData filteredLabelDatabase = this->subTableByKeys( aLabelDatabase, intersectKeys );
	QStringList labelGroups = this->labelGroups( filteredLabelDatabase, aLabelIndex );

	// Initialize the table
	lpmldata::TabularData foldConfigTableTemporal;
	QStringList headerNamesTemporal;
	for ( int i = 0; i < aMaximumFoldCount; ++i )
	{
		headerNamesTemporal.push_back( "Fold-" + QString::number( i + 1 ) );
	}
	foldConfigTableTemporal.setHeader( headerNamesTemporal );

	// Fill up the table with empty values.
	for ( auto key : intersectKeys )
	{
		QVariantList vector;
		for ( int i = 0; i < aMaximumFoldCount; ++i )
		{
			vector.push_back( 0 );
		}
		foldConfigTableTemporal.insert( key, vector );
	}

	// Build up a labelgroup key map.
	QMap< int, QStringList > labelGroupKeys;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		labelGroupKeys.insert( i, this->keysByLabelGroup( filteredLabelDatabase, aLabelIndex, labelGroups.at( i ) ) );
	}

	//Determine minority sample count, take at least 1 sample.
	int minorityGroupSize = filteredLabelDatabase.rowCount();  // This is higher than the maximum, unless we have only one label outcome.
	int minorityGroupIndex = -1;
	QVector< int > originalSamplecounts;

	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		originalSamplecounts.push_back( labelGroupKeys.value( i ).size() );

		if ( labelGroupKeys.value( i ).size() < minorityGroupSize )
		{
			minorityGroupSize = labelGroupKeys.value( i ).size();
			minorityGroupIndex = i;
		}
	}

	int minorityGroupSizeValdiation = std::max( 1, int( minorityGroupSize * aSubsetRatio ) );  // This is needed to make sure that at least 1 element is chosen from the minority group for validation.

	// Determine all other sample counts relative to the minority sample counts.
	QVector< int > subsetSampleCountsValidation;
	QVector< int > subsetSampleCountsTraining;

	subsetSampleCountsValidation.resize( labelGroupKeys.size() );
	subsetSampleCountsTraining.resize( labelGroupKeys.size() );

	int trainingSize = minorityGroupSize - minorityGroupSizeValdiation;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		subsetSampleCountsValidation[ i ] = minorityGroupSizeValdiation;
		subsetSampleCountsTraining[ i ] = trainingSize;
	}

	//Fill up the foldconfig table.
	/*QVector< int > labelGroupIndices;
	labelGroupIndices.resize( labelGroupKeys.size() );
	labelGroupIndices.fill( 0 );*/

	int foldIndex = 0;
	while ( foldIndex < aMaximumFoldCount )
	//for ( int foldIndex = 0; foldIndex < aMaximumFoldCount; ++foldIndex )
	{
		qDebug() << "Generating fold " << foldIndex;

		QMap< QString, int > currentFoldConfig;

		for ( int i = 0; i < labelGroups.size(); ++i )
		{
			QStringList keys = labelGroupKeys.value( i );

			// Shuffle the keys.
			std::shuffle( keys.begin(), keys.end(), mRandomGenerator );

			// Take out validation samples.
			int offset = 0;
			for ( int sampleIndex = 0; sampleIndex < subsetSampleCountsValidation.at( i ); ++sampleIndex )
			{
				QString validationKey = keys.at( sampleIndex );
				if ( !currentFoldConfig.contains( validationKey ) )
				{
					currentFoldConfig.insert( validationKey, 1 );
				}
				else
				{
					qDebug() << "Validation key " << validationKey << "is already in the foldConfig...";
				}
			}

			// Take out training samples.
			offset += subsetSampleCountsValidation.at( i );
			for ( int sampleIndex = offset; sampleIndex < subsetSampleCountsTraining.at( i ) + offset; ++sampleIndex )
			{
				QString trainingKey = keys.at( sampleIndex );
				if ( !currentFoldConfig.contains( trainingKey ) )
				{
					currentFoldConfig.insert( trainingKey, 0 );
				}
				else
				{
					qDebug() << "Training key " << trainingKey << "is already in the foldConfig...";
				}
			}

			// Sign the rest of the samples as non-selected.
			offset += subsetSampleCountsTraining.at( i );
			for ( int sampleIndex = offset; sampleIndex < keys.size(); ++sampleIndex )
			{
				QString leftoverKey = keys.at( sampleIndex );
				if ( !currentFoldConfig.contains( leftoverKey ) )
				{
					currentFoldConfig.insert( leftoverKey, -1 );
				}
				else
				{
					qDebug() << "Leftover key " << leftoverKey << "is already in the foldConfig...";
				}
			}

			//// Select validation samples.
			//int foundValidationCount = 0;
			//while ( true )
			//{
			//	QString validationKey = keys.at( labelGroupIndices.at( i ) );
			//	if ( !currentFoldConfig.contains( validationKey ) )
			//	{
			//		currentFoldConfig.insert( validationKey, 1 );
			//		++foundValidationCount;
			//		++labelGroupIndices[ i ];
			//		if ( labelGroupIndices.at( i ) == keys.size() )
			//		{
			//			labelGroupIndices[ i ] = 0;
			//		}
			//	}
			//	if ( foundValidationCount == subsetSampleCountsValidation.at( i ) )
			//	{
			//		break;
			//	}
			//}

			//// Select training samples.
			//int foundTrainingCount = 0;
			//for ( auto key : keys )
			//{
			//	if ( !currentFoldConfig.contains( key ) )
			//	{
			//		currentFoldConfig.insert( key, 0 );
			//		++foundTrainingCount;
			//	}
			//	if ( foundTrainingCount == subsetSampleCountsTraining.at( i ) )
			//	{
			//		break;
			//	}
			//}
			//
			//// Sign the rest of the keys as non-selected.
			//for ( auto key : keys )
			//{
			//	if ( !currentFoldConfig.contains( key ) )
			//	{
			//		currentFoldConfig.insert( key, -1 );
			//	}
			//}
		}

		// Check if the given fold config has ever been created before. If yes, stop the fold creation.		
		QStringList foldConfigKeys = foldConfigTableTemporal.keys();
		bool isDuplicateFoldConfig = false;
		for ( int columnIndex = 0; columnIndex < foldConfigTableTemporal.columnCount(); ++columnIndex )
		{
			isDuplicateFoldConfig = false;
			bool isDuplicateFoldConfigActual = true;
			for ( int rowIndex = 0; rowIndex < foldConfigKeys.size(); ++rowIndex )
			{
				QString key = foldConfigKeys.at( rowIndex );
				int savedfoldValue = foldConfigTableTemporal.valueAt( key, columnIndex ).toInt();
				int currentFoldValue = currentFoldConfig.value( key );
				if ( savedfoldValue != currentFoldValue )
				{
					isDuplicateFoldConfigActual = false;
				}
			}

			if ( isDuplicateFoldConfigActual )
			{
				isDuplicateFoldConfig = true;
				qDebug() << "Duplicate found!";
				break;
			}
		}

		// Duplicate found?
		if ( isDuplicateFoldConfig )  // Stop the fold generation.
		{
			continue;
		}
		else  // Save the new fold config in the table.
		{
			for ( auto key : currentFoldConfig.keys() )
			{
				foldConfigTableTemporal.valueAt( key, foldIndex ) = currentFoldConfig.value( key );
			}
			++foldIndex;
		}
	}

	// Initialize the final fold config table
	lpmldata::TabularData foldConfigTable;
	QStringList headerNames;
	for ( int i = 0; i < aMaximumFoldCount; ++i )
	{
		headerNames.push_back( "Fold-" + QString::number( i + 1 ) );
	}
	foldConfigTable.setHeader( headerNames );

	// Fill up the table with values.
	for ( auto key : intersectKeys )
	{
		QVariantList vector;
		for ( int i = 0; i < aMaximumFoldCount; ++i )
		{
			vector.push_back( foldConfigTableTemporal.valueAt( key, i ).toInt() );
		}
		foldConfigTable.insert( key, vector );
	}

	return foldConfigTable;
}

//-----------------------------------------------------------------------------

QMap< int, QStringList > TabularDataFilter::stratifiedKFoldMap( int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex )
{
	QMap< int, QStringList > foldSelectionKeys;

	QStringList intersectKeys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	// Take the intersection of the database entries based on the actual labelIndex.
	lpmldata::TabularData filteredLabelDatabase = this->subTableByKeys( aLabelDatabase, intersectKeys );

	// Determine the label groups.
	QStringList labelGroups = this->labelGroups( filteredLabelDatabase, aLabelIndex );

	//Determine minority sample count, take at least 1 sample.
	int minorityGroupSize = filteredLabelDatabase.rowCount();  // This is higher than the maximum, unless we have only one label outcome.
	int minorityGroupIndex = -1;
	QVector< int > originalSamplecounts;

	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		QStringList labelGroupKeys = this->keysByLabelGroup( filteredLabelDatabase, aLabelIndex, labelGroups.at( i ) );
		originalSamplecounts.push_back( labelGroupKeys.size() );

		if ( labelGroupKeys.size() < minorityGroupSize )
		{
			minorityGroupSize = labelGroupKeys.size();
			minorityGroupIndex = i;
		}
	}

	int newMinorityGroupSize = std::max( 1, int( minorityGroupSize * aSubsetRatio ) );  // This is needed to make sure that at least 1 element is chosen from the minority group.
	double correctedSubsetRatio = double( newMinorityGroupSize ) / double( minorityGroupSize );  // Corrected subset ratio based on corrected minority subset size.

	// Determine all other sample counts relative to the minority sample counts.
	QVector< int > subsetSampleCounts;
	subsetSampleCounts.resize( labelGroups.size() );
	subsetSampleCounts[ minorityGroupIndex ] = newMinorityGroupSize;
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		if ( i != minorityGroupIndex )  // Check all groups except the minority one.
		{
			subsetSampleCounts[ i ] = originalSamplecounts.at( i ) * correctedSubsetRatio;
		}
	}

	// Go through labelgroup keys, shuffle them and take out the samples.
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		QStringList labelGroupKeys = this->keysByLabelGroup( filteredLabelDatabase, aLabelIndex, labelGroups.at( i ) );
		
		std::shuffle( labelGroupKeys.begin(), labelGroupKeys.end(), mRandomGenerator );

		int globalLabelGroupIndex = 0;
		for ( int foldIndex = 0; foldIndex < aFoldCount; ++foldIndex )
		{
			// Take new samples.
			for ( int sampleIndex = 0; sampleIndex < subsetSampleCounts.at( i ); ++sampleIndex )
			{
				foldSelectionKeys[ foldIndex ].push_back( labelGroupKeys.at( globalLabelGroupIndex ) );
				++globalLabelGroupIndex;

				// This is a naive approach, later on we have to filter out redundant columns!
				if ( globalLabelGroupIndex == labelGroupKeys.size() )
				{
					std::shuffle( labelGroupKeys.begin(), labelGroupKeys.end(), mRandomGenerator );
					globalLabelGroupIndex = 0;
				}
			}			
		}
	}

	return foldSelectionKeys;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::normalize( lpmldata::TabularData& aFeatureDatabase )
{
	QVector< double > mins = aFeatureDatabase.mins();
	QVector< double > maxs = aFeatureDatabase.maxs();

	lpmldata::TabularData normalizedTable;
	normalizedTable.header() = aFeatureDatabase.header();

	QStringList keys = aFeatureDatabase.keys();

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString key = keys.at( keyIndex );
		QVariantList featureVector = aFeatureDatabase.value( key );
		QVariantList normalizedFeatureVector;

		for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
		{
			double normalizedFeature = ( featureVector.at( columnIndex ).toDouble() - mins.at( columnIndex ) ) / ( maxs.at( columnIndex ) - mins.at( columnIndex ) );
			normalizedFeatureVector.push_back( normalizedFeature );
		}

		normalizedTable.insert( key, normalizedFeatureVector );
	}

	return normalizedTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByMask( const lpmldata::TabularData& aFeatureDatabase, const QVector< double >& aFeatureMask )
{
	lpmldata::TabularData subTable;
	lpmldata::TabularDataHeader originalHeader = aFeatureDatabase.header();
	QStringList originalHeaderKeys = originalHeader.keys();
	lpmldata::TabularDataHeader subTableHeader;
	QString type = "Float";

	// Build up subtable header
	int headerIndex = 0;
	for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
	{
		if ( aFeatureMask.at( columnIndex ) > 0.0 )
		{
			//auto value = originalHeader.value( originalHeaderKeys.at( columnIndex ) );
			auto value = originalHeader.value( QString::number( columnIndex ) );
			subTableHeader.insert( QString::number( headerIndex ), value );
			++headerIndex;
		}
	}
	
	subTable.header() = subTableHeader;
	
	QStringList keys = aFeatureDatabase.keys();

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString key = keys.at( keyIndex );
		QVariantList featureVector = aFeatureDatabase.value( key );
		QVariantList subTableFeatureVector;

		for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
		{
			if ( aFeatureMask.at( columnIndex ) > 0.0 )
			{
				subTableFeatureVector.push_back( featureVector.at( columnIndex ) );
			}	
		}

		subTable.insert( key, subTableFeatureVector );
	}

	return subTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByFeatureNames( const lpmldata::TabularData& aFeatureDatabase, const QStringList& aFeatureNames )
{
	lpmldata::TabularData subTable;
	lpmldata::TabularDataHeader originalHeader = aFeatureDatabase.header();
	QStringList originalHeaderKeys = originalHeader.keys();
	lpmldata::TabularDataHeader subTableHeader;
	QString type = "Float";
	QVector< int > featureMask;

	// Build up subtable header
	int headerIndex = 0;
	for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
	{
		auto headerValue = originalHeader.value( QString::number( columnIndex ) );
		QString value = headerValue.toStringList().at( 0 );

		if ( aFeatureNames.contains( value ) )
		{
			subTableHeader.insert( QString::number( headerIndex ), value );
			featureMask.push_back( columnIndex );
			++headerIndex;
		}	
	}

	subTable.header() = subTableHeader;

	QStringList keys = aFeatureDatabase.keys();

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString key = keys.at( keyIndex );
		QVariantList featureVector = aFeatureDatabase.value( key );
		QVariantList subTableFeatureVector;

		for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
		{
			if ( featureMask.contains( columnIndex ) )
			{
				subTableFeatureVector.push_back( featureVector.at( columnIndex ) );
			}
		}

		subTable.insert( key, subTableFeatureVector );
	}

	return subTable;
}

//-----------------------------------------------------------------------------

QVector< int > TabularDataFilter::maskIndicesByNames( const lpmldata::TabularData& aFeatureDatabase, const QStringList& aFeatureNames )
{
	lpmldata::TabularDataHeader originalHeader = aFeatureDatabase.header();
	QStringList originalHeaderKeys = originalHeader.keys();
	QVector< int > maskIndices;

	// Select indices of feature names in the feature database.
	for ( unsigned int columnIndex = 0; columnIndex < aFeatureDatabase.columnCount(); ++columnIndex )
	{
		auto headerValue = originalHeader.value( QString::number( columnIndex ) );
		QString value = headerValue.toStringList().at( 0 );

		if ( aFeatureNames.contains( value ) )
		{
			maskIndices.push_back( columnIndex );
		}
	}

	return maskIndices;
}

//-----------------------------------------------------------------------------


// Measure distance of two vectors by selected features
double TabularDataFilter::distance( const QVariantList& aFirstVector, const QVariantList& aSecondVector, QVector< double > aFeatureMask )
{
	double distance = 0.0;

	if ( !aFeatureMask.isEmpty() && aFeatureMask.size() == aFirstVector.size() && aFirstVector.size() == aSecondVector.size() )  // Valid mask & sizes are OK?
	{
		for ( int i = 0; i < aFirstVector.size(); ++i )
		{
			if ( aFeatureMask.at( i ) > 0.0 )
			{
				double first = aFirstVector.at( i ).toDouble();
				double second = aSecondVector.at( i ).toDouble();

				distance += std::pow( first - second, 2.0 );
			}
		}
	}
	else if ( aFirstVector.size() == aSecondVector.size() ) // No mask and sizes are OK?
	{
		for ( int i = 0; i < aFirstVector.size(); ++i )
		{
			double first = aFirstVector.at( i ).toDouble();
			double second = aSecondVector.at( i ).toDouble();

			distance += std::pow( first - second, 2.0 );
		}
	}
	else  // Sizes do not match.
	{
		return -1.0;
	}

	return std::sqrt( distance );
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::nearestNeighbors( const QString& aKey, lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask )
{
	QVariantList firstVector = aFeatureDatabase.value( aKey );
	QStringList keys = aFeatureDatabase.keys();
	QMap< double, QString > neighbors;

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		if ( aKey != actualKey )
		{
			QVariantList secondVector = aFeatureDatabase.value( actualKey );
			double distance = this->distance( firstVector, secondVector, aFeatureMask );
			neighbors.insertMulti( distance, actualKey );
		}
	}

	QList< double > neighborDistances = neighbors.keys();
	QStringList nearestNeighborKeys;

	for ( int neighborIndex = 0; neighborIndex < std::min( aNeighborCount, neighborDistances.size() ); ++neighborIndex )
	{
		nearestNeighborKeys.push_back( neighbors.value( neighborDistances.at( neighborIndex ) ) );
	}

	return nearestNeighborKeys;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::nearestNeighbors( const QVariantList& aFeatureVector, lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask )
{
	QStringList keys = aFeatureDatabase.keys();
	QMap< double, QString > neighbors;

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		
		QVariantList secondVector = aFeatureDatabase.value( actualKey );
		double distance = this->distance( aFeatureVector, secondVector, aFeatureMask );
		neighbors.insertMulti( distance, actualKey );
	}

	QList< double > neighborDistances = neighbors.keys();
	QStringList nearestNeighborKeys;

	for ( int neighborIndex = 0; neighborIndex < std::min( aNeighborCount, neighborDistances.size() ); ++neighborIndex )
	{
		nearestNeighborKeys.push_back( neighbors.value( neighborDistances.at( neighborIndex ) ) );
	}

	return nearestNeighborKeys;
}

//-----------------------------------------------------------------------------

QMap< QString, QStringList > TabularDataFilter::nearestNeighborMap( lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask )
{
	QStringList keys = aFeatureDatabase.keys();
	QMap< QString, QStringList > nearestNeighborMap;

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		nearestNeighborMap.insert( actualKey, this->nearestNeighbors( actualKey, aFeatureDatabase, aNeighborCount, aFeatureMask ) );
	}

	return nearestNeighborMap;
}

//-----------------------------------------------------------------------------

QMap< QString, double > TabularDataFilter::scoreFeaturesByMSMOTE( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aNeighborCount, QVector< double > aFeatureMask )
{
	QMap< QString, double > MSMOTECategories;
	QStringList keys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	QMap< QString, QStringList > neighborMap = nearestNeighborMap( subTableByKeys( aFeatureDatabase, keys ), aNeighborCount, aFeatureMask );
	QStringList labelGroups = this->labelGroups( aLabelDatabase, aLabelIndex );

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		QVector< int > labelOccurrences;
		labelOccurrences.resize( labelGroups.size() );
		labelOccurrences.fill( 0 );

		QStringList neighbors = neighborMap.value( actualKey );
		QString labelOfActualKey = aLabelDatabase.value( actualKey ).at( aLabelIndex ).toString();

		// Go through neighbors.
		for ( int neighborIndex = 0; neighborIndex < neighbors.size(); ++neighborIndex )
		{
			// Identify the label value of the neighbor.
			QString neighborKey = neighbors.at( neighborIndex );
			QString neighborLabel = aLabelDatabase.value( neighborKey ).at( aLabelIndex ).toString();

			// Save the label occurrence into labelOccurrences.
			labelOccurrences[ labelGroups.indexOf( neighborLabel ) ]++;
		}

		// Score the actualKey based on labelOccurrences;
		int labelIndexOfActualKey = labelGroups.indexOf( labelOfActualKey );

		int othersOccurrences = 0;
		int actualOccurrences = 0;
		int allOccurrences = 0;
		int maxOccurrences = 0;

		for ( int i = 0; i < labelOccurrences.size(); ++i )
		{
			if ( labelOccurrences.at( i ) > maxOccurrences )  // Identify the maximum occurrence
			{
				maxOccurrences = labelOccurrences.at( i );
			}

			if ( i == labelIndexOfActualKey )  // We found the occurrence of the actual key
			{
				actualOccurrences = labelOccurrences.at( i );
			}
			else
			{
				othersOccurrences += labelOccurrences.at( i );
			}

			allOccurrences += labelOccurrences.at( i );  // Collect all occurrences.
		}

		double occurrenceScore = ( double( actualOccurrences ) / double( maxOccurrences ) ) - ( double( othersOccurrences ) / ( double( maxOccurrences ) * double( labelOccurrences.size() - 1 ) ) );
		MSMOTECategories.insert( actualKey, occurrenceScore );
	
	}

	return MSMOTECategories;
}

//-----------------------------------------------------------------------------

QMap< QString, QString > TabularDataFilter::categorizeMSMOTE( const QMap< QString, double >& aMSMOTEScores, double aOutlierThreshold, double aSafeThreshold )
{
	QMap< QString, QString > MSMOTECategories;

	QStringList keys = aMSMOTEScores.keys();

	for ( int i = 0; i < keys.size(); ++i )
	{
		QString key = keys.at( i );
		double value = aMSMOTEScores.value( key );

		if ( value < aOutlierThreshold ) // Outlier
		{
			MSMOTECategories.insert( key, "Outlier" );
		}
		else if ( value > aSafeThreshold ) // Safe
		{
			MSMOTECategories.insert( key, "Safe" );
		}
		else  // Borderline
		{
			MSMOTECategories.insert( key, "Borderline" );
		}
	}

	return MSMOTECategories;
}

//-----------------------------------------------------------------------------

int TabularDataFilter::MSMOTECategoryCount( const QMap< QString, QString >& aMSMOTECategories, QString aCategoryType, QStringList aReferenceKeys )
{
	int categoryCount = 0;
	QStringList keys;

	if ( aReferenceKeys.isEmpty() )
	{
		keys = aMSMOTECategories.keys();
	}
	else
	{
		keys = aReferenceKeys;
	}

	for ( int i = 0; i < keys.size(); ++i )
	{
		QString key = keys.at( i );
		QString value = aMSMOTECategories.value( key );

		if ( value == aCategoryType )
		{
			++categoryCount;
		}
	}

	return categoryCount;
}

//-----------------------------------------------------------------------------

// MSMOTE table creation
QPair< lpmldata::TabularData, lpmldata::TabularData> TabularDataFilter::MSMOTE( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const ulint aSampleCount, QVector< double > aFeatureMask )
{
	lpmldata::TabularData MSMOTEdFDB;// = aFeatureDatabase;
	lpmldata::TabularData MSMOTEdLDB;// = aLabelDatabase;
	lpmldata::TabularData aFeatureDatabaseFiltered = aFeatureDatabase;

	std::uniform_real_distribution< double > realRand( -0.01, 0.01 );

	MSMOTEdFDB.header() = aFeatureDatabase.header();
	MSMOTEdLDB.header() = aLabelDatabase.header();

	QStringList keys = this->commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );
	lpmldata::TabularData FDBNormalized = this->normalize( aFeatureDatabase );
	

	// Identify the largest group and the minority groups.
	int largestLabelCount = 0;
	int largestLabelIndex = 0;
	QStringList labelGroups = this->labelGroups( aLabelDatabase, aLabelIndex );
	for ( int labelIndex = 0; labelIndex < labelGroups.size(); ++labelIndex )
	{
		QStringList keysOfLabel = this->keysByLabelGroup( aLabelDatabase, aLabelIndex, labelGroups.at( labelIndex ) );
		if ( keysOfLabel.size() > largestLabelCount )
		{
			largestLabelCount = keysOfLabel.size();
			largestLabelIndex = labelIndex;
		}
	}

	// Identify outliers in the whole dataset.
	QMap< QString, double > MSMOTEMap = scoreFeaturesByMSMOTE( FDBNormalized, aLabelDatabase, aLabelIndex, 5, aFeatureMask );
	QMap< QString, QString > MSMOTECategories = categorizeMSMOTE( MSMOTEMap, 0.7, 0.9 );

	// Delete the outliers in the largest group.
	QStringList keysOfLargestGroup = this->keysByLabelGroup( aLabelDatabase, aLabelIndex, labelGroups.at( largestLabelIndex ) );
	for ( int i = 0; i < keysOfLargestGroup.size(); ++i )
	{
		QString key = keysOfLargestGroup.at( i );
		if ( MSMOTECategories.value( key ) == "Outlier" )
		{
			aFeatureDatabaseFiltered.remove( key );
		}
	}

	// TODO: Repeat the scoring on the filtered database, consider it as reference for MSMOTE scoring.
	FDBNormalized = this->normalize( aFeatureDatabaseFiltered );

	// Go through all groups and fill them up with new samples till aSampleCount is reached.
	int MSMOTEdFeatureCounter = -1;

	for ( int labelIndex = 0; labelIndex < labelGroups.size(); ++labelIndex )
	{
		//if ( labelIndex != largestLabelIndex )  // Minority group
		{
			QStringList keysOfMinority = this->keysByLabelGroup( aLabelDatabase, aLabelIndex, labelGroups.at( labelIndex ) );

			int maximumNeighborCount = std::max( 2, std::min( 5, keysOfMinority.size() / 5 ) );
			MSMOTEMap = scoreFeaturesByMSMOTE( FDBNormalized, aLabelDatabase, aLabelIndex, maximumNeighborCount, aFeatureMask );

			// Generate MSMOTE categories.
			MSMOTECategories = categorizeMSMOTE( MSMOTEMap, -0.1, 0.5 );
			QStringList MSMOTECategoriesKeys = MSMOTECategories.keys();
			
			// Filter out keys of the minority group
			QStringList commonKeys = keysOfMinority.toSet().intersect( MSMOTECategoriesKeys.toSet() ).toList();
			int outlierCount = MSMOTECategoryCount( MSMOTECategories, "Outlier", commonKeys );
			int safeCount = MSMOTECategoryCount( MSMOTECategories, "Safe", commonKeys );
			int borderlineCount = MSMOTECategoryCount( MSMOTECategories, "Borderline", commonKeys );

			QStringList commonKeysfiltered;
			for ( int fk = 0; fk < commonKeys.size(); ++fk )
			{
				QString actKey = commonKeys.at( fk );
				if ( MSMOTECategories.value( actKey ) != "Outlier" )
				{
					commonKeysfiltered.push_back( actKey );
				}
			}

			commonKeys = commonKeysfiltered;

			lpmldata::TabularData tableOfMinority = subTableByKeys( aFeatureDatabase, commonKeys );

			int newSampleCounter = 0;
			bool isNewSamplesNeeded = true;

			int localBSNeighborcount = 1;

			while ( isNewSamplesNeeded )
			{
				++localBSNeighborcount;

				for ( int i = 0; i < commonKeys.size(); ++i )
				{
					if ( isNewSamplesNeeded == false ) break;

					QString key = commonKeys.at( i );
					
					QVariantList keyFeature = aFeatureDatabase.value( key );
					QVariantList keyLabel = aLabelDatabase.value( key );

					MSMOTEdFDB.insert( key, keyFeature );
					MSMOTEdLDB.insert( key, keyLabel );
					++newSampleCounter;

					QString category = MSMOTECategories.value( key );
					//QMap< QString, double > neighborScores;

					QStringList neighbors = nearestNeighbors( key, tableOfMinority, localBSNeighborcount, aFeatureMask );
					for ( int j = 0; j < neighbors.size(); ++j )
					{
						QString neighborKey = neighbors.at( j );							
						QVariantList neighborFeature = aFeatureDatabase.value( neighborKey ); // Take the neighbor feature vector.
						//double neighborScore = MSMOTEMap.value( neighborKey );  // Take the MSMOTEMap value.
						double neighborDistance = this->distance( keyFeature, neighborFeature, aFeatureMask );  // Calculate the distance from the given key.
						//double fitnessScore = neighborScore; // / neighborDistance;  // Come up with a summarized score to characterize fitness. Larger neighborScore and smaller neighborDistance is larger fitness!
						//neighborScores.insert( neighborKey, neighborDistance );
					}

					// Weighted average the N neighbors to create the new feature variant.
					QVector < double > vecvar;
					vecvar.resize( aFeatureDatabase.columnCount() );
					vecvar.fill( 0.0 );

					
					// Store the point itself which we are investigating.
					for ( int f = 0; f < vecvar.size(); ++f )
					{
						vecvar[ f ] = keyFeature.at( f ).toDouble();
					}

					// Add the neighbor points with a little +random value.
					double sumWeights = 1.0;  // We need to count the weight of the actual keyFeature.
					for ( int j = 0; j < neighbors.size(); ++j )
					{
						QString neighborKey = neighbors.at( j );
						QVariantList neighborVariant = aFeatureDatabase.value( neighborKey );

						double weight = ( 1.0 + realRand( mRandomGenerator ) );
						sumWeights += weight;

						for ( int f = 0; f < neighborVariant.size(); ++f )
						{
							vecvar[ f ] += neighborVariant.at( f ).toDouble() * weight;
						}
					}

					// Normalize by sumWeight.
					QVariantList featureVariant;
					
					for ( int f = 0; f < vecvar.size(); ++f )
					{
						featureVariant.push_back( vecvar.at( f ) / sumWeights );
					}

					// Create the label variant. NA all others that are irrelevant.
					QVariantList labelVariant = aLabelDatabase.value( key );
					QString originalLabel = labelVariant.at( aLabelIndex ).toString();
					QVector< QVariant > labelVecVar = labelVariant.toVector();
					labelVecVar.fill( "NA" );
					labelVariant = labelVecVar.toList();
					labelVariant[ aLabelIndex ] = originalLabel;

					// Add the new feature-label pair to the database pairs.

					++MSMOTEdFeatureCounter;
					++newSampleCounter;

					MSMOTEdFDB.insert( "MSMOTE-" + QString::number( MSMOTEdFeatureCounter ), featureVariant );
					MSMOTEdLDB.insert( "MSMOTE-" + QString::number( MSMOTEdFeatureCounter ), labelVariant );

					if ( newSampleCounter == aSampleCount ) isNewSamplesNeeded = false;
					
				}
			}
		}
	}
	
	QPair< lpmldata::TabularData, lpmldata::TabularData> MSMOTEdTables;
	MSMOTEdTables.first = MSMOTEdFDB;
	MSMOTEdTables.second = MSMOTEdLDB;

	return MSMOTEdTables;
}


//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByLabelGroup( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel )
{
	QStringList keys = commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );  // Determine common keys.

	// Construct the subset tabular data.
	lpmldata::TabularData subset;
	subset.header() = aFeatureDatabase.header();

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		QString actualLabel = aLabelDatabase.valueAt( actualKey, aLabelIndex ).toString(); // Read out the label from the column of aLabelDatabase deterlined by the aLabelIndex.

		if ( actualLabel == aReferenceLabel )  // We found a label matching the reference label.
		{
			subset.insert( actualKey, aFeatureDatabase.value( actualKey ) );  // Save the given row.
		}
	}

	return subset;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::keysByLabelGroup( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel )
{
	QStringList keys = aLabelDatabase.keys();
	QStringList keysByLabel;

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		QString actualLabel = aLabelDatabase.valueAt( actualKey, aLabelIndex ).toString(); // Read out the label from the column of aLabelDatabase deterlined by the aLabelIndex.

		if ( actualLabel == aReferenceLabel )  // We found a label matching the reference label.
		{
			keysByLabel.push_back( actualKey );
		}
	}

	return keysByLabel;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::bootstrap( const lpmldata::TabularData& aFeatureDatabase, const ulint aSampleCount, bool aIsOriginalSamplesPreserved )
{
	lpmldata::TabularData bootstrapTable;  // This is the final table to contain the means of all bootstraps generated below.
	bootstrapTable.header() = aFeatureDatabase.header();
	QStringList keys = aFeatureDatabase.keys();
	lint originalSampleCount = keys.size();  // Determine the number of samples in the original database.

	if ( originalSampleCount == 0 ) // The input feature database is empty!
	{
		return bootstrapTable;
	}

	int residualSampleCount = 0;
	if ( aIsOriginalSamplesPreserved )
	{
		bootstrapTable = aFeatureDatabase;
		residualSampleCount = aSampleCount - originalSampleCount;
	}
	else
	{
		residualSampleCount = aSampleCount;
	}
	
	if ( residualSampleCount <= 0 )  // Still place for bootstrapped samples?
	{
		return bootstrapTable;
	}

	std::uniform_int_distribution<ulint>  intDistribution( 0, originalSampleCount - 1 );  // Real distribution between intervals.
	omp_set_nested( 0 );

#pragma omp parallel for
	for ( lint sampleIndex = 0; sampleIndex < residualSampleCount; ++sampleIndex ) // Iterate aSampleCount times and create bootstraps
	{
		// Create a bootstrap to contain the randomly selected samples.
		lpmldata::TabularData actualBootstrap;
		actualBootstrap.header() = aFeatureDatabase.header();

		for ( ulint rowIndex = 0; rowIndex < originalSampleCount; ++rowIndex )  // Go through the input database and randomly select rows
		{
			ulint randomRowIndex = intDistribution( mRandomGenerator );  // Pick a random number.
			actualBootstrap.insert( QString::number( rowIndex ), aFeatureDatabase.value( keys.at( randomRowIndex ) ) );  // Since the same row can be sleected multiple times, we need to generate each row a unique ID.
		}

		QString sampleKey = "BS-" + QString::number( sampleIndex );
		QVariantList sampleMeans = actualBootstrap.means();

#pragma omp critical
		bootstrapTable.insert( sampleKey, sampleMeans );  // Store the mean of the bootstrapped table.

	}

	return bootstrapTable;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::bootstrapN( const lpmldata::TabularData& aFeatureDatabase )
{
	lpmldata::TabularData bootstrappedTable;  // This is the final table to contain the means of all bootstraps generated below.
	bootstrappedTable.header() = aFeatureDatabase.header();
	QStringList keys = aFeatureDatabase.keys();

	std::shuffle( mAbc.begin(), mAbc.end(), mRandomGenerator );
	
	std::uniform_int_distribution<ulint>  intDistribution( 0, keys.size() - 1 );  // Real distribution between intervals.

	// Create a bootstrap to contain the randomly selected samples.
	for ( ulint rowIndex = 0; rowIndex < keys.size(); ++rowIndex )  // Go through the input database and randomly select rows
	{
		ulint randomRowIndex = intDistribution( mRandomGenerator );  // Pick a random number.
		QString newKey = QString::number( rowIndex ) + "-" + QString::number( randomRowIndex ) + "-" + mAbc.left( 8 );
		bootstrappedTable.insert( newKey, aFeatureDatabase.value( keys.at( randomRowIndex ) ) );  // Since the same row can be sleected multiple times, we need to generate each row a unique ID.
	}

	return bootstrappedTable;
}

//-----------------------------------------------------------------------------

QPair< lpmldata::TabularData, lpmldata::TabularData> TabularDataFilter::bootstrapN( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase )
{
	lpmldata::TabularData bootstrappedFDB;  // This is the final table to contain the means of all bootstraps generated below.
	lpmldata::TabularData bootstrappedLDB;  // This is the final table to contain the means of all bootstraps generated below.
	bootstrappedFDB.header() = aFeatureDatabase.header();
	bootstrappedLDB.header() = aLabelDatabase.header();

	QStringList FDBkeys = aFeatureDatabase.keys();
	QStringList LDBkeys = aLabelDatabase.keys();
	QStringList keys = FDBkeys.toSet().intersect( LDBkeys.toSet() ).toList();

	std::shuffle( mAbc.begin(), mAbc.end(), mRandomGenerator );

	std::uniform_int_distribution<ulint>  intDistribution( 0, keys.size() - 1 );  // Real distribution between intervals.

	// Create a bootstrap to contain the randomly selected samples.
	for ( ulint rowIndex = 0; rowIndex < keys.size(); ++rowIndex )  // Go through the input database and randomly select rows
	{
		ulint randomRowIndex = intDistribution( mRandomGenerator );  // Pick a random number.
		QString newKey = keys.at( randomRowIndex ) + "-" + QString::number( rowIndex ) + "-" + mAbc.left( 8 );
		bootstrappedFDB.insert( newKey, aFeatureDatabase.value( keys.at( randomRowIndex ) ) );  // Since the same row can be sleected multiple times, we need to generate each row a unique ID.
		bootstrappedLDB.insert( newKey,   aLabelDatabase.value( keys.at( randomRowIndex ) ) );
	}

	return QPair < lpmldata::TabularData, lpmldata::TabularData >( bootstrappedFDB, bootstrappedLDB );
}

//-----------------------------------------------------------------------------

QPair< lpmldata::TabularData, lpmldata::TabularData > TabularDataFilter::bootstrap( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const ulint aSampleCount, bool aIsOriginalSamplesPreserved )
{
	lpmldata::TabularData featureDatabaseBootstrapped;  // Bootstrapped feature database.
	lpmldata::TabularData labelDatabaseBootstrapped;  // Bootstrapped label database.

	featureDatabaseBootstrapped.header() = aFeatureDatabase.header();  // Copy the header from the original database.
	labelDatabaseBootstrapped.header() = aLabelDatabase.header();  // Copy the header from the original database.

	QStringList labelGroupList = labelGroups( aLabelDatabase, aLabelIndex );  // Take the possible label outcomes for the given label index.

	ulint globalUniqueIndex = 0;

	// Go through label groups, take aSampleCount amount of samples with bootstrapping and fill in a new feature as well as label database.
	for ( int labelGroupIndex = 0; labelGroupIndex < labelGroupList.size(); ++labelGroupIndex )
	{
		int residualSampleCount = 0;
		// Create a table containing only feature vectors of the given label group entry.
		lpmldata::TabularData tableByLabelGroup = subTableByLabelGroup( aFeatureDatabase, aLabelDatabase, aLabelIndex, labelGroupList.at( labelGroupIndex ) );
		QStringList originalKeys = tableByLabelGroup.keys();

		// Here filtering out the outliers from the given labelgroup could happen.

		// Add original values;
		if ( aIsOriginalSamplesPreserved )
		{
			residualSampleCount = aSampleCount - originalKeys.size();

			if ( originalKeys.size() == 0 ) continue;
			for ( int sampleIndex = 0; sampleIndex < originalKeys.size(); ++sampleIndex )
			{
				QString originalKey = originalKeys.at( sampleIndex );
				featureDatabaseBootstrapped.insert( originalKey, aFeatureDatabase.value( originalKey ) );
				labelDatabaseBootstrapped.insert( originalKey, aLabelDatabase.value( originalKey ) );
			}
		}
		else
		{
			residualSampleCount = aSampleCount;
		}
		

		// Create a bootstrap table containing aSampleCount - original count samples.
		lpmldata::TabularData bootstrappedTable = bootstrap( tableByLabelGroup, residualSampleCount, false );
		QStringList bootstrappedKeys = bootstrappedTable.keys();

		// Create the label variant. NA all others that are irrelevant.
		QVariantList labelVariantBS;
		QString originalLabel = labelGroupList.at( labelGroupIndex );
		QVector< QVariant > labelVecVar = labelVariantBS.toVector();
		labelVecVar.resize( aLabelDatabase.columnCount() );
		labelVecVar.fill( "NA" );
		labelVariantBS = labelVecVar.toList();
		labelVariantBS[ aLabelIndex ] = originalLabel;

		for ( int sampleIndex = 0; sampleIndex < bootstrappedKeys.size(); ++sampleIndex )
		{
			QString bootstrappedKey = bootstrappedKeys.at( sampleIndex );
			featureDatabaseBootstrapped.insert( "BS-" + QString::number( globalUniqueIndex ), bootstrappedTable.value( bootstrappedKey ) );
			labelDatabaseBootstrapped.insert( "BS-" + QString::number( globalUniqueIndex ), labelVariantBS );
			globalUniqueIndex++;
		}
	}

	return QPair < lpmldata::TabularData, lpmldata::TabularData >( featureDatabaseBootstrapped, labelDatabaseBootstrapped );
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByUniqueness( const lpmldata::TabularData& aFeatureDatabase )
{
	QStringList keys = aFeatureDatabase.keys();
	QStringList redundantKeys;

	lpmldata::TabularData uniqueFeatures = aFeatureDatabase;

	for ( int firstKeyIndex = 0; firstKeyIndex < keys.size() - 1; ++firstKeyIndex )
	{
		QVariantList first = aFeatureDatabase.value( keys.at( firstKeyIndex ) );

		for ( int secondKeyIndex = firstKeyIndex + 1; secondKeyIndex < keys.size(); ++secondKeyIndex )
		{
			QVariantList second = aFeatureDatabase.value( keys.at( secondKeyIndex ) );

			if ( isEqual( first, second ) )  // We found a redundant feature vector.
			{
				redundantKeys.push_back( keys.at( secondKeyIndex ) );
			}
		}
	}

	for ( int redundantKeyIndex = 0; redundantKeyIndex < redundantKeys.size(); ++redundantKeyIndex )
	{
		uniqueFeatures.remove( redundantKeys.at( redundantKeyIndex ) );
	}

	return uniqueFeatures;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::labelGroups( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
{
	QStringList labelsOfDatabase;
	QStringList keys = aLabelDatabase.keys();

	for ( int rowIndex = 0; rowIndex < keys.size(); ++rowIndex )
	{
		QString actualLabel = aLabelDatabase.valueAt( keys.at( rowIndex ), aLabelIndex ).toString();

		if ( actualLabel == "NA" && aIsNAIncluded == false ) continue;

		if ( !labelsOfDatabase.contains( actualLabel ) )
		{
			labelsOfDatabase.push_back( actualLabel );
		}
	}

	qSort( labelsOfDatabase );

	return labelsOfDatabase;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByKeys( const lpmldata::TabularData& aTabularData, QStringList aReferenceKeys )
{
	lpmldata::TabularData subsetByKeyTabularData;
	subsetByKeyTabularData.header() = aTabularData.header();

	QStringList tabularDataKeys = aTabularData.keys();
	QStringList commonKeys = tabularDataKeys.toSet().intersect( aReferenceKeys.toSet() ).toList();  // Determine common keys.

	for ( int keyIndex = 0; keyIndex < commonKeys.size(); ++keyIndex )
	{
		QString actualKey = commonKeys.at( keyIndex );
		QVariantList row = aTabularData.value( actualKey );
		subsetByKeyTabularData.insert( actualKey, row );
	}

	return subsetByKeyTabularData;
}

//-----------------------------------------------------------------------------

//QStringList TabularDataFilter::commonKeys( const muwdata::TabularData& aFeatureDatabase, const muwdata::TabularData& aLabelDatabase )
//{
//	return aFeatureDatabase.keys().toSet().intersect( aLabelDatabase.keys().toSet() ).toList();
//}

//-----------------------------------------------------------------------------


QStringList TabularDataFilter::commonKeys( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
{
	QSet< QString > featureKeys = aFeatureDatabase.keys().toSet();
	QSet< QString > labelKeys = aLabelDatabase.keys().toSet();
	QStringList NAKeys = keysByLabelGroup( aLabelDatabase, aLabelIndex, "NA" );

	if ( !NAKeys.isEmpty() && !aIsNAIncluded )
	{
		labelKeys.subtract( NAKeys.toSet() );
	}

	return featureKeys.intersect( labelKeys ).toList();
}

//-----------------------------------------------------------------------------

QVector< QMap< QVariant, double > > TabularDataFilter::equalizedBagging( lpmldata::DataPackage* aDataPackage, const unsigned int aNumberBags, const double aBagFraction )
{
	QVector< QMap< QVariant, double > > keysToWeights;

	for ( int i = 0; i < aNumberBags; ++i )
	{
		QMap< QVariant, double > bagMap;

		for ( auto uniqueLabel : aDataPackage->labelOutcomes() )
		{
			auto labelKeys = keysByLabelGroup( aDataPackage->labelDatabase(), aDataPackage->labelIndex(), uniqueLabel );

			std::uniform_int_distribution< unsigned int > intDistribution( 0, labelKeys.size() - 1 );

			int baggedSubsetsize = std::ceil( double( aDataPackage->sampleKeys().size() * aBagFraction ) / double( aDataPackage->labelOutcomes().size() ) );

			for ( int j = 0; j < baggedSubsetsize; ++j )
			{
				int randomIndex = intDistribution( mRandomGenerator );
				auto randomKey = labelKeys.at( randomIndex );
				if ( !bagMap.contains( randomKey ) )
				{
					bagMap.insert( randomKey, 1.0 );
				}
				else
				{
					bagMap[ randomKey ] += 1.0;
				}
			}
		}

		keysToWeights.append( bagMap );
	}



	//// Determine indices of samples from different label groups
	//QMap< QVariant, QVector< int > > labelGroupIndices;
	//int sampleIndex = 0;
	//for ( auto key : aDataPackage->sampleKeys() )
	//{
	//	for ( auto uniqueLabel : aDataPackage->labelOutcomes() )
	//	{
	//		if ( aDataPackage->labelDB().valueAt( key, aDataPackage->labelIndex() ) == uniqueLabel )	labelGroupIndices[ uniqueLabel ].append( sampleIndex );
	//	}
	//	sampleIndex++;
	//}

	//std::random_device rd;					// Obtain a random number from hardware.
	//std::mt19937 randomGenerator( rd() );	// Seed the generator.
	//double numberUniqueLabels = aDataPackage->labelOutcomes().size();
	//double numberSamples      = aDataPackage->sampleKeys().size();

	//QVector< QMap< QVariant, double > > keysToWeights;

	//for ( int bagIndex = 0; bagIndex < aNumberBags; bagIndex++ )
	//{
	//	QMap< QVariant, double > bagMap;
	//	for ( auto labelGroup : labelGroupIndices.keys() )
	//	{
	//		QVector< int > randomSampleIndices;
	//		for ( int randomNumberIndex = 0; randomNumberIndex < ceil( ( numberSamples / numberUniqueLabels ) * aBagFraction ); randomNumberIndex++ )
	//		{
	//			std::uniform_int_distribution< unsigned int >  intDistribution( 0, labelGroupIndices[ labelGroup ].size() - 1 );
	//			randomSampleIndices.push_back( intDistribution( randomGenerator ) );  // Pick a random number.
	//		}

	//		for ( auto sampleIndex : randomSampleIndices )
	//		{
	//			QString key = aDataPackage->sampleKeys()[ labelGroupIndices[ labelGroup ][ sampleIndex ] ];

	//			bool keyExists;
	//			if ( bagMap.keys().contains( key ) )	keyExists = true;

	//			if ( keyExists )	bagMap[ key ] += 1;
	//			else				bagMap[ key ] = 1;
	//		}
	//	}

	//	keysToWeights.append( bagMap );
	//}

	return keysToWeights;
}

//-----------------------------------------------------------------------------

QVector< QMap< QVariant, double > > TabularDataFilter::bagging( lpmldata::DataPackage* aDataPackage, const unsigned int aNumberBags, const double aBagFraction )
{
	// Determine indices of samples from different label groups
	QMap< QVariant, QVector< int > > labelGroupIndices;
	int sampleIndex = 0;
	for ( auto key : aDataPackage->labelDatabase().keys() )
	{
		for ( auto uniqueLabel : aDataPackage->labelOutcomes() )
		{
			if ( aDataPackage->labelDatabase().valueAt( key, aDataPackage->labelIndex() ) == uniqueLabel )	labelGroupIndices[ uniqueLabel ].append( sampleIndex );
		}
		sampleIndex++;
	}

	std::random_device rd;
	std::mt19937 randomGenerator( rd() );
	double numberSamples = aDataPackage->labelDatabase().rowCount();

	QVector< QMap< QVariant, double > > keysToWeights;

	for ( int bagIndex = 0; bagIndex < aNumberBags; bagIndex++ )
	{
		QMap< QVariant, double > bagMap;

		QVector< int > randomSampleIndices;
		for ( int randomNumberIndex = 0; randomNumberIndex < numberSamples * aBagFraction; randomNumberIndex++ )
		{
			std::uniform_int_distribution< unsigned int >  intDistribution( 0, numberSamples - 1 );
			randomSampleIndices.push_back( intDistribution( randomGenerator ) );
		}

		for ( auto sampleIndex : randomSampleIndices )
		{
			QString key = aDataPackage->labelDatabase().keys()[ sampleIndex ];

			bool keyExists;
			if ( bagMap.keys().contains( key ) )	keyExists = true;

			if ( keyExists )	bagMap[ key ] += 1;
			else				bagMap[ key ] = 1;
		}

		keysToWeights.append( bagMap );
	}

	return keysToWeights;
}

//-----------------------------------------------------------------------------

QVector< QMap< QVariant, double > > TabularDataFilter::walkerBagging( lpmldata::TabularData& aLabelSet, unsigned int aNumberBags, const double aBagFraction )
{
	QVector< QMap< QVariant, double > > keysToWeights;

	while ( aNumberBags > 0 )
	{
		QMap< QVariant, double > bag;
		int samplesPerBag = floor( aLabelSet.rowCount() * aBagFraction );

		if ( samplesPerBag < aLabelSet.rowCount() )
		{
			bag = resampleWithWeights( aLabelSet, bag, false );
		}
		else
		{
			bag = resampleWithWeights( aLabelSet, bag, true );
		}

		keysToWeights.append( bag );
		aNumberBags--;
	}

	return keysToWeights;
}

//-----------------------------------------------------------------------------

QMap< QVariant, double > TabularDataFilter::resampleWithWeights( lpmldata::TabularData& aLabelSet, QMap< QVariant, double >& aBagWeights, bool aRepresentUsingWeights )
{
	QMap< QVariant, double > initialWeights;
	for ( auto key : aLabelSet.keys() )
	{
		initialWeights[ key ] = 1.0;
	}

	// Walker's method, pp. 232 "Stochastic Simulation" by B.D. Ripley
	QVector< double > P( QVector< double >( aLabelSet.rowCount() ) );
	QVector< double > Q( QVector< double >( aLabelSet.rowCount() ) );

	QVector< int > A( QVector< int >( aLabelSet.rowCount() ) );
	QVector< int > W( QVector< int >( aLabelSet.rowCount() ) );

	P = initialWeights.values().toVector();

	normalize( P );

	int M = initialWeights.size();
	int NN = -1;
	int NP = M;
	for ( int I = 0; I < M; I++ )
	{
		if ( P[ I ] < 0 )
		{
			qDebug() << "Error: Weights have to be positive.";
		}
		Q[ I ] = M * P[ I ];
		if ( Q[ I ] < 1.0 )
		{
			W[ ++NN ] = I;
		}
		else
		{
			W[ --NP ] = I;	// If no weight smaller 1, W holds descending indices of samples
		}
	}

	if ( NN > -1 && NP < M ) 	// Will not be executed if all weights >= 1
	{
		for ( int S = 0; S < M - 1; S++ )
		{
			int I = W[ S ];
			int J = W[ NP ];
			A[ I ] = J;
			Q[ J ] += Q[ I ] - 1.0;
			if ( Q[ J ] < 1.0 )
			{
				NP++;
			}
			if ( NP >= M )
			{
				break;
			}
		}
	}

	for ( int I = 0; I < M; I++ )
	{
		Q[ I ] += I;	// Q holds descending sample indices
	}

	QVector< int > counts( M );		// Counts how often an index is randomly chosen

	std::random_device rd;
	std::mt19937 randomGenerator( rd() );
	std::uniform_real_distribution<>  realDistribution( 0, 1 );

	for ( int i = 0; i < aLabelSet.rowCount(); i++ )
	{
		int ALRV;
		double randomFactor = realDistribution( randomGenerator );
		double U = M * randomFactor;
		int I = floor( U );		// Random index

		if ( U < Q[ I ] )
		{
			ALRV = I;
		}
		else
		{
			ALRV = A[ I ];	// only applied if weight smaller 1
		}
		if ( aRepresentUsingWeights )
		{
			counts[ ALRV ]++;
		}
		else
		{
			aBagWeights[ initialWeights.keys()[ ALRV ] ] = initialWeights[ initialWeights.keys()[ ALRV ] ];
		}
		if ( !aRepresentUsingWeights )
		{
			aBagWeights[ initialWeights.keys()[ aLabelSet.rowCount() - 1 ] ] = 1;
		}
	}

	// Add data based on counts if weights should represent numbers of copies.
	if ( aRepresentUsingWeights )
	{
		for ( int i = 0; i < counts.size(); i++ )
		{
			if ( counts[ i ] > 0 )
			{
				aBagWeights[ initialWeights.keys()[ i ] ] = counts[ i ];
			}
		}
	}

	return aBagWeights;
}

//-----------------------------------------------------------------------------

void TabularDataFilter::normalize( QVector< double >& aVector )
{
	double vectorSum = 0;
	for ( auto value : aVector )
	{
		vectorSum += value;
	}

	for ( int index = 0; index < aVector.size(); index++ )
	{
		aVector[ index ] /= vectorSum;
	}
}

//----------------------------------------------------------------------------

lpmldata::TabularData TabularDataFilter::subTableByAttributes( lpmldata::TabularData& aFeatureSet, const QList< int >& aAttributeIndices )
{
	lpmldata::TabularData subset;

	// Create header
	lpmldata::TabularDataHeader header;
	int newColumnIndex = 0;
	for ( auto index : aAttributeIndices )
	{
		QString value = aFeatureSet.headerNames()[ index ];
		QString type = "Float";
		QVariantList headerValue = { value, type };
		//header.insert( QString::number( index ), headerValue );
		header.insert( QString::number( newColumnIndex ), headerValue );
		newColumnIndex++;
	}
	subset.header() = header;

	// Add samples with attribute value subset
	for ( auto key : aFeatureSet.keys() )
	{
		QVariantList attributeSubset;

		for ( auto index : aAttributeIndices )
		{
			attributeSubset.append( aFeatureSet.valueAt( key, index ) );
		}

		subset.insert( key, attributeSubset );
	}

	return subset;
}

//-----------------------------------------------------------------------------

}
