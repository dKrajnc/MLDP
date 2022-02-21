#include <DataRepresentation/DataPackage.h>
//#include <Evaluation/TabularDataFilter.h>

namespace lpmldata
{

//-----------------------------------------------------------------------------

DataPackage::DataPackage( lpmldata::TabularData& aFDB, lpmldata::TabularData& aLDB, QString aLabelName, QStringList aIncludedKeys )
{
	mIncludedKeys = aIncludedKeys;
	initialize( aFDB, aLDB, aLabelName );
}

//-----------------------------------------------------------------------------

void DataPackage::setActiveLabel( QString aLabel )
{
	QList< QString > labels = mLDB.headerNames();

	if ( labels.contains( aLabel ) )
	{
		mActiveLabelIndex = labels.indexOf( aLabel );
	}
	else
	{
		qDebug() << "DataPackage - Error: There is no " << aLabel << " in " << labels;
	}
}

//-----------------------------------------------------------------------------

const QStringList DataPackage::labelGroups() const
{
	QStringList labelsOfDatabase;
	QStringList keys = mLDB.keys();

	for ( int rowIndex = 0; rowIndex < keys.size(); ++rowIndex )
	{
		QString actualLabel = mLDB.valueAt( keys.at( rowIndex ), mActiveLabelIndex ).toString();

		if ( actualLabel == "NA" ) continue;

		if ( !labelsOfDatabase.contains( actualLabel ) )
		{
			labelsOfDatabase.push_back( actualLabel );
		}
	}

	qSort( labelsOfDatabase );

	return labelsOfDatabase;
}

//-----------------------------------------------------------------------------

QStringList DataPackage::commonKeys() const
{
	QStringList featureKeys = mFDB.keys();
	QStringList labelKeys   = mLDB.keys();
	QStringList labelKeysNA;

	for ( int i = 0; i < labelKeys.size(); ++i )
	{
		auto value = mLDB.valueAt( labelKeys.at( i ), mActiveLabelIndex ).toString();

		if ( value == "NA" )
		{
			labelKeysNA.push_back( labelKeys.at( i ) );
		}
	}

	auto featureKeySet = featureKeys.toSet();
	auto labelKeySet   = labelKeys.toSet();
	auto labelKeySetNA = labelKeysNA.toSet();

	return featureKeySet.intersect( labelKeySet ).subtract( labelKeySetNA ).toList();
}

//-----------------------------------------------------------------------------
//Used in: FeatureSelection
lpmldata::TabularData DataPackage::featureDatabaseSubset( QStringList aFeatureNames ) const
{
	lpmldata::TabularData filteredFDB;
	filteredFDB.setHeader( aFeatureNames );

	auto originalFeatureNames = mFDB.headerNames();
	auto keys                 = mFDB.keys();

	QVector< int > indices;
	
	for ( auto& featureName : aFeatureNames )
	{
		for ( int i = 0; i < originalFeatureNames.size(); ++i )
		{
			auto originalName = originalFeatureNames.at( i );
			auto featureIndex = -1;

			
			if ( featureName == originalName )
			{
				featureIndex = i;
				indices.push_back( featureIndex );
			}			
		}
	}

	for ( auto& key : keys )
	{
		QVariantList filteredFeatureVector;
		
		for ( int i = 0; i < indices.size(); ++i )
		{
			filteredFeatureVector.push_back( mFDB.valueAt( key, indices.at( i ) ) );
		}

		filteredFDB.insert( key, filteredFeatureVector );
	}	

	return filteredFDB;
}

//-----------------------------------------------------------------------------
//Used in IsolationForest, TomekLinks, RandomUndersampling
lpmldata::TabularData DataPackage::labelDatabaseSubset( const QStringList& aKeys ) const
{
	lpmldata::TabularData filteredLDB;
	auto activeLabel = mLDB.headerNames().at( 0 );
	auto header      = mLDB.headerNames();
	filteredLDB.setHeader( header );


	for ( int i = 0; i < aKeys.size(); ++i )
	{
		for ( auto& key : aKeys )
		{
			auto labelValues = mLDB.value( key );

			filteredLDB.insert( key, labelValues );
		}
	}

	return filteredLDB;
}

//-----------------------------------------------------------------------------
//Used in: Oversampling
lpmldata::TabularData DataPackage::labelDatabaseSubset( QStringList& aSynthNames, int& aMinorityLabel ) const
{
	lpmldata::TabularData filteredLDB;	
	auto activeLabel = mLDB.headerNames().at( 0 );
	filteredLDB.setHeader( activeLabel );
	

	auto originalKeys = commonKeys();
	auto originalLabelNames = mLDB.headerNames();
	
	for ( int i = 0; i < originalKeys.size(); ++i )
	{
		auto key = originalKeys.at( i );	
		QVariantList originalValues;

		for ( int j = 0; j < originalLabelNames.size(); ++j )
		{
			auto originalFeatureName = originalLabelNames.at( j );
			if ( activeLabel == originalFeatureName )
			{
				originalValues.push_back( mLDB.valueAt( key, j ) );
			}
		}

		filteredLDB.insert( key, originalValues );
	}

	for ( int i = 0; i < aSynthNames.size(); ++i )
	{
		auto key = aSynthNames.at( i );
		QVariantList synthValues;
		synthValues.push_back( aMinorityLabel );

		filteredLDB.insert( key, synthValues );
	}

	return filteredLDB;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataPackage::syntheticLabelDatabaseSubset( const QList< QPair< QString, double > >& aSyntheticPairList ) const
{
	lpmldata::TabularData syntheticLDB;
	auto activeLabel = mLDB.headerNames().at( 0 );
	syntheticLDB.setHeader( activeLabel );

	for ( int i = 0; i < aSyntheticPairList.size(); ++i )
	{
		auto key   = aSyntheticPairList.at( i ).first;
		auto label = aSyntheticPairList.at( i ).second;
		QVariantList synthLabel;
		synthLabel.push_back( label );

		syntheticLDB.insert( key, synthLabel );
	}

	return syntheticLDB;
}

//-----------------------------------------------------------------------------

//Used in PCA
lpmldata::TabularData DataPackage::featureDatabaseSubset( QVector< QVector< double > >& aFeatureValues, QStringList& aFeatureNames ) const
{
	lpmldata::TabularData filteredFDB;	

	filteredFDB.setHeader( aFeatureNames );
	
	auto keys = commonKeys();	
	std::sort( keys.begin(), keys.end() );


	for ( int i = 0; i < keys.size(); ++i )
	{
		auto key = keys.at( i );
		
		QVariantList filteredFeatureVector;

		for ( int j = 0; j < aFeatureNames.size(); ++j )
		{

			auto featureValues = aFeatureValues.at( j );

			filteredFeatureVector.push_back( featureValues.at( i ) );

		}

		filteredFDB.insert( key, filteredFeatureVector );
	}

	return filteredFDB;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataPackage::featureDatabaseSubset( QVector< QVector< double > >& aFeatureColumns ) const
{
	lpmldata::TabularData generatedFDB;

	auto keys    = mFDB.keys();
	auto headers = mFDB.headerNames();
	generatedFDB.setHeader( headers );
	

	for ( int i = 0; i < aFeatureColumns.at( 0 ).size(); ++i )
	{
		QVariantList featureRow;
		
		for ( int j = 0; j < aFeatureColumns.size(); ++j )
		{
			QVariant value = aFeatureColumns.at( j ).at( i ); //double to Qvariant implict conversion
			
			featureRow.push_back( value );
		}

		auto key = keys.at( i );
		generatedFDB.insert( key, featureRow );
	}

	return generatedFDB;
}

//-----------------------------------------------------------------------------

//Used in: IsolationForest, TomekLinks, RandomUndersampling
lpmldata::TabularData DataPackage::sampleDatabaseSubset( const QStringList& aKeys ) const
{
	lpmldata::TabularData filteredFDB;

	auto headers = mFDB.headerNames();
	filteredFDB.setHeader( headers );

	for ( auto& key : aKeys )
	{
		auto featureRow = mFDB.value( key );

		filteredFDB.insert( key, featureRow );
	}
	
	return filteredFDB;
}

//-----------------------------------------------------------------------------
//Used in: Oversampling
lpmldata::TabularData DataPackage::sampleDatabaseSubset( QMap< QString, QVector< double > >& aSynthSamples ) const
{
	lpmldata::TabularData filteredFDB;
	auto headers = mFDB.headerNames();
	filteredFDB.setHeader( headers );

	auto keys = commonKeys();

	for ( int i = 0; i < keys.size(); ++i )
	{
		QVariantList featureRow;

		for ( auto& key : keys )
		{
			featureRow = mFDB.value( key );

			filteredFDB.insert( key, featureRow );
		}
	}

	for ( int i = 0; i < aSynthSamples.size(); ++i )
	{
		auto key           = aSynthSamples.keys().at( i );
		auto featureVector = aSynthSamples.value( key );		
		filteredFDB.insert( key, toQVariantList( featureVector ) );
	}

	return filteredFDB;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataPackage::syntheticSampleDatabaseSubset( QMap< QString, QVector< double > >& aSynthSamples ) const
{
	lpmldata::TabularData syntheticFDB;
	auto header = mFDB.headerNames();
	syntheticFDB.setHeader( header );

	for ( int i = 0; i < aSynthSamples.size(); ++i )
	{
		auto key           = aSynthSamples.keys().at( i );
		auto featureVector = aSynthSamples.value( key );
		syntheticFDB.insert( key, toQVariantList( featureVector ) );
	}

	return syntheticFDB;
}

//-----------------------------------------------------------------------------

QStringList DataPackage::keysOfLabelGroup( QString aLabelGroup ) const
{	
	QStringList keysOfLabelGroup;
	QStringList commonKeys = this->commonKeys();

	for ( int i = 0; i < commonKeys.size(); ++i )
	{		
		QString actualLabel = mLDB.valueAt( commonKeys.at( i ), mActiveLabelIndex ).toString();

		if ( actualLabel == "NA" ) continue;

		if ( actualLabel == aLabelGroup )
		{
			keysOfLabelGroup.push_back( commonKeys.at( i ) );
		}					
	}

	qSort( keysOfLabelGroup );

	return keysOfLabelGroup;
}

//-----------------------------------------------------------------------------

QMap< QString, QStringList > DataPackage::keysOfLabelGroups() const
{
	QMap< QString, QStringList > keysOfLabelgroups;
	QStringList labelGroups = this->labelGroups();
	
	
	for ( int i = 0; i < labelGroups.size(); ++i )
	{
		keysOfLabelgroups.insert( labelGroups.at( i ), keysOfLabelGroup( labelGroups.at( i ) ) );
	}

	return keysOfLabelgroups;

}

//-----------------------------------------------------------------------------

QMap< QString, double >  DataPackage::distance( const QMap< QString, QVariantList >& aFirst, const QMap< QString, QVariantList >& aSecond ) const
{
	QMap< QString, double > dist;
	QString names;

	double squared            = 0.0;
	double substractedSquared = 0.0;


	auto firstValues  = aFirst.values().at( 0 );
	auto secondValues = aSecond.values().at( 0 );

	for ( int i = 0; i < aFirst.keys().size(); ++i )
	{

		for ( int j = 0; j < firstValues.size(); ++j )
		{
			auto secondNames = aSecond.keys().at( i );
			auto firstNames = aFirst.keys().at( i );

			substractedSquared = pow( secondValues.at( j ).toDouble() - firstValues.at( j ).toDouble(), 2.0 );
			substractedSquared += substractedSquared;
			names = secondNames;
		}

	}
	squared = sqrt( substractedSquared );

	dist.insert( names, squared );

	return dist;
}

//-----------------------------------------------------------------------------

double DataPackage::distance( QVector< double >& aFirst, QVector< double >& aSecond )
{
	double dist               = 0.0;
	double substractedSquared = 0.0;

	for ( int i = 0; i < aFirst.size(); ++i )
	{
		substractedSquared = pow( aSecond.at( i ) - aFirst.at( i ), 2.0 );
		substractedSquared += substractedSquared;
	}
	dist = sqrt( substractedSquared );

	return dist;
}

//-----------------------------------------------------------------------------

double DataPackage::distance( const QVariantList& aFirst, const QVariantList& aSecond ) const
{
	double dist               = 0.0;
	double substractedSquared = 0.0;

	for ( int i = 0; i < aFirst.size(); ++i )
	{
		substractedSquared = pow( aSecond.at( i ).toDouble() - aFirst.at( i ).toDouble(), 2.0 );
		substractedSquared += substractedSquared;
	}
	dist = sqrt( substractedSquared );

	return dist;
}

//-----------------------------------------------------------------------------

//lpmldata::TabularData DataPackage::normalize( const lpmldata::TabularData& aFDB ) const
//{
//	QVector< QVector< double > > normalizedFeatureColumns;
//	for ( unsigned int i = 0; i < aFDB.columnCount(); ++i )
//	{
//		QVector< double > normalzedFeatureColumn;
//
//		auto featureColumn = aFDB.column( i ).toVector();
//		auto columnSize    = featureColumn.size();
//		double sum         = 0.0;
//
//		std::for_each( featureColumn.begin(), featureColumn.end(),
//			[ &sum ]( QVariant x ) { sum += x.toDouble(); } );
//
//		auto mean              = this->mean( sum, columnSize );
//		auto standardDeviation = this->standardDeviation( featureColumn, mean );
//
//		for ( auto& element : featureColumn )
//		{
//			double normalizedElement = ( element.toDouble() - mean ) / standardDeviation;
//
//			normalzedFeatureColumn.push_back( normalizedElement );
//		}
//		normalizedFeatureColumns.push_back( normalzedFeatureColumn );
//	}
//
//	auto normalizedFDB = featureDatabaseSubset( normalizedFeatureColumns );
//
//	return normalizedFDB;
//}

//-----------------------------------------------------------------------------

double DataPackage::mean( const double& aSum, const int& aColumnSize ) const
{
	return aSum / aColumnSize;
}

//-----------------------------------------------------------------------------

double DataPackage::standardDeviation( const QVector< QVariant >& aFeatureColumn, const double& aMean ) const
{
	double deviation = 0;
	double sum       = 0;
	
	std::for_each( aFeatureColumn.begin(), aFeatureColumn.end(),
		[ &sum, &aMean ]( QVariant x ) { auto squaredDifference = std::pow( x.toDouble() - aMean, 2 ); sum += squaredDifference; } );

	deviation = std::sqrt( sum );

	return deviation;
}

//-----------------------------------------------------------------------------

double DataPackage::standardDeviation( const QVector< double >& aFeatureColumn, const double& aMean ) const
{
	double deviation = 0;
	double sum       = 0;

	std::for_each( aFeatureColumn.begin(), aFeatureColumn.end(),
		[ &sum, &aMean ]( double x ) { auto squaredDifference = std::pow( x - aMean, 2 ); sum += squaredDifference; } );

	deviation = std::sqrt( sum );

	return deviation;
}

//-----------------------------------------------------------------------------

QVector< double > DataPackage::featureColumn( QString aFeatureKey ) const
{
	QVector< double > featureValues;
	auto headers = mFDB.headerNames();
	auto keys    = mFDB.keys();

	//Find feature index 
	auto index   = 0;

	for (int i = 0; i < headers.size(); ++i )
	{	
		auto name = headers.at( i );

		if ( name == aFeatureKey )
		{
			index = i;
		}		
	}

	//Get values 
	for ( auto& key : keys )
	{
		auto value = mFDB.valueAt( key, index );
		featureValues.push_back( value.toDouble() );
	}

	return featureValues;
}

//-----------------------------------------------------------------------------

QList< int > DataPackage::labels( QString aLabelName ) const
{
	QList< int > labelValues;
	auto headers = mLDB.headerNames();
	auto keys    = mLDB.keys();

	//Find label index 
	auto index = 0;

	for ( int i = 0; i < headers.size(); ++i )
	{
		auto name = headers.at( i );

		if ( name == aLabelName )
		{
			index = i;
		}
	}

	//Get values 
	for ( auto& key : keys )
	{
		auto value = mLDB.valueAt( key, index );
		labelValues.push_back( value.toDouble() );
	}

	return labelValues;
}

//-----------------------------------------------------------------------------

QVector< double > DataPackage::normalizeFeature( const QVector< double >& aFeatureColumn ) const
{
	QVector< double > normalzedFeatureColumn;
	double sum      = 0.0;
	auto columnSize = aFeatureColumn.size();

	std::for_each( aFeatureColumn.begin(), aFeatureColumn.end(),
		[ &sum ]( double x ) { sum += x; } );

	auto mean              = this->mean( sum, columnSize );
	auto standardDeviation = this->standardDeviation( aFeatureColumn, mean );

	for ( auto& element : aFeatureColumn )
	{
		double normalizedElement = ( element - mean ) / standardDeviation;

		normalzedFeatureColumn.push_back( normalizedElement );
	}

	return normalzedFeatureColumn;
}

//-----------------------------------------------------------------------------

QVariantList DataPackage::toQVariantList( const QVector<double>& aVector ) const
{
	QVariantList list;

	for ( auto& element : aVector )
	{
		list.push_back( element );
	}

	return list;
}

//-----------------------------------------------------------------------------

void DataPackage::initialize( lpmldata::TabularData& aFDB, lpmldata::TabularData& aLDB, QString aLabelName )
{
	mFDB       = aFDB;
	mLDB       = aLDB;
	mLabelName = aLabelName;
	

	if ( !mIncludedKeys.isEmpty() )
	{
		mFDB = subTableByKeys( mFDB, mIncludedKeys );
		mLDB = subTableByKeys( mLDB, mIncludedKeys );
	}

	eraseIncompleteRecords( mFDB );


	mFeatureCount          = mFDB.columnCount();
	QStringList labelNames = mLDB.headerNames();
	mLabelIndex            = labelNames.indexOf( mLabelName );

	if ( mLabelIndex == -1 )
	{
		qDebug() << "DataPackage ERROR: Label name " << mLabelName << "is not part of label names: " << labelNames;
		system( "pause" );
		mIsValidDataset = false;
		return;
	}

	mLabelOutcomes = labelGroups( mLDB, mLabelIndex );
	mSampleKeys    = commonKeys( mFDB, mLDB, mLabelIndex );

	/*if ( mLabelOutcomes.size() < 2 )
	{
		qDebug() << "DataPackage ERROR: Insufficient number of label outcomes: " << mLabelOutcomes;
		system( "pause" );
		mIsValidDataset = false;
		return;
	}*/


	mIsValidDataset = true;
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataPackage::subTableByKeys( const lpmldata::TabularData& aTabularData, QStringList aReferenceKeys )
{
	lpmldata::TabularData subsetByKeyTabularData;
	subsetByKeyTabularData.header() = aTabularData.header();

	QStringList tabularDataKeys = aTabularData.keys();
	QStringList commonKeys      = tabularDataKeys.toSet().intersect( aReferenceKeys.toSet() ).toList();  // Determine common keys.

	for ( int keyIndex = 0; keyIndex < commonKeys.size(); ++keyIndex )
	{
		QString actualKey = commonKeys.at( keyIndex );
		QVariantList row  = aTabularData.value( actualKey );

		subsetByKeyTabularData.insert( actualKey, row );
	}

	return subsetByKeyTabularData;
}

//-----------------------------------------------------------------------------

void DataPackage::eraseIncompleteRecords( lpmldata::TabularData& aFeatureDatabase )
{
	QStringList keysToDelete;

	for ( auto key : aFeatureDatabase.keys() )
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

QStringList DataPackage::labelGroups( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
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

QStringList DataPackage::commonKeys( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
{
	QSet< QString > featureKeys = aFeatureDatabase.keys().toSet();
	QSet< QString > labelKeys   = aLabelDatabase.keys().toSet();
	QStringList NAKeys          = keysByLabelGroup( aLabelDatabase, aLabelIndex, "NA" );

	if ( !NAKeys.isEmpty() && !aIsNAIncluded )
	{
		labelKeys.subtract( NAKeys.toSet() );
	}
	
	return featureKeys.intersect( labelKeys ).toList();
}

//-----------------------------------------------------------------------------

QStringList DataPackage::keysByLabelGroup( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel )
{
	QStringList keysByLabel;

	for ( int keyIndex = 0; keyIndex < aLabelDatabase.keys().size(); ++keyIndex )
	{
		QString actualKey   = aLabelDatabase.keys().at( keyIndex );
		QString actualLabel = aLabelDatabase.valueAt( actualKey, aLabelIndex ).toString(); // Read out the label from the column of aLabelDatabase deterlined by the aLabelIndex.

		if ( actualLabel == aReferenceLabel )  // We found a label matching the reference label.
		{
			keysByLabel.push_back( actualKey );
		}
	}

	return keysByLabel;
}

//-----------------------------------------------------------------------------

bool DataPackage::isBalanced()
{	
	if ( keysOfLabelGroups().size() == 2 )
	{
		double majorityCount = getMajorityCount();
		double minorityCount = getMinorityCount();
		
		//Calculate percentage difference
		int absoluteSizeDifference  = std::abs( majorityCount - minorityCount );
		double average              = ( majorityCount + minorityCount ) / 2;
		double percentageDifference = ( absoluteSizeDifference / average ) * 100;

		if ( percentageDifference < 20 ) //Set treshold to 20% for imbalance difference
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		qDebug() << "Error - not a binary classification!";
		std::system( EXIT_SUCCESS );	
	}
}

//-----------------------------------------------------------------------------

int DataPackage::minorityCount() const
{
	QMap< QString, int > classSizes;

	for ( int i = 0; i < keysOfLabelGroups().size(); ++i )
	{
		auto key           = keysOfLabelGroups().keys().at( i );
		auto samplesNumber = keysOfLabelGroups().value( key );
		int size           = samplesNumber.size();

		classSizes.insert( key, size );
	}

	//Binary classification
	return std::min( classSizes.values().at( 0 ), classSizes.values().at( 1 ) );	
}

//-----------------------------------------------------------------------------

int DataPackage::majorityCount() const
{
	QMap< QString, int > classSizes;

	for ( int i = 0; i < keysOfLabelGroups().size(); ++i )
	{
		auto key           = keysOfLabelGroups().keys().at( i );
		auto samplesNumber = keysOfLabelGroups().value( key );
		int size           = samplesNumber.size();

		classSizes.insert( key, size );
	}

	//Binary classification
	return std::max( classSizes.values().at( 0 ), classSizes.values().at( 1 ) );
}

//-----------------------------------------------------------------------------

int DataPackage::sampleCountOfPercentage( const double& aValidationPercentage ) const
{
	auto totalSampleCount      = mFDB.keys().size();	
	auto validationSampleCount = ( totalSampleCount * aValidationPercentage ) / 100;

	return (int)validationSampleCount;
}

//-----------------------------------------------------------------------------

int DataPackage::getMinorityIndex() const
{
	int minorityIndex;

	if ( keysOfLabelGroups().values().at( 0 ).size() < keysOfLabelGroups().values().at( 1 ).size() )
	{
		auto minorityLabel = keysOfLabelGroups().keys().at( 0 );
		minorityIndex      = keysOfLabelGroups().keys().indexOf( minorityLabel );
	}
	else
	{
		auto minorityLabel = keysOfLabelGroups().keys().at( 1 );
		minorityIndex      = keysOfLabelGroups().keys().indexOf( minorityLabel );
	}

	return minorityIndex;
}

//-----------------------------------------------------------------------------

int DataPackage::getMinorityCount() const
{
	int minorityCount;


	if ( keysOfLabelGroups().values().at( 0 ).size() < keysOfLabelGroups().values().at( 1 ).size() )
	{
		minorityCount = keysOfLabelGroups().values().at( 0 ).size();
	}
	else
	{
		minorityCount = keysOfLabelGroups().values().at( 1 ).size();
	}

	return minorityCount;
}

//-----------------------------------------------------------------------------

int DataPackage::getMinorityLabel() const
{
	int minorityLabel;

	if ( keysOfLabelGroups().values().at( 0 ).size() < keysOfLabelGroups().values().at( 1 ).size() )
	{
		minorityLabel = keysOfLabelGroups().keys().at( 0 ).toInt();
	}
	else
	{
		minorityLabel = keysOfLabelGroups().keys().at( 1 ).toInt();
	}


	return minorityLabel;
}

//-----------------------------------------------------------------------------

int DataPackage::getMajorityIndex() const
{
	int majorityIndex;

	if ( keysOfLabelGroups().values().at( 0 ).size() > keysOfLabelGroups().values().at( 1 ).size() )
	{
		auto minorityLabel = keysOfLabelGroups().keys().at( 0 );
		majorityIndex      = keysOfLabelGroups().keys().indexOf( minorityLabel );
	}
	else
	{
		auto minorityLabel = keysOfLabelGroups().keys().at( 1 );
		majorityIndex      = keysOfLabelGroups().keys().indexOf( minorityLabel );
	}

	return majorityIndex;
}

//-----------------------------------------------------------------------------

int DataPackage::getMajorityCount() const
{
	int majorityCount;


	if ( keysOfLabelGroups().values().at( 0 ).size() > keysOfLabelGroups().values().at( 1 ).size() )
	{
		majorityCount = keysOfLabelGroups().values().at( 0 ).size();
	}
	else
	{
		majorityCount = keysOfLabelGroups().values().at( 1 ).size();
	}

	return majorityCount;
}

//-----------------------------------------------------------------------------

int DataPackage::getMajorityLabel() const
{
	int majorityLabel;

	if ( keysOfLabelGroups().values().at( 0 ).size() > keysOfLabelGroups().values().at( 1 ).size() )
	{
		majorityLabel = keysOfLabelGroups().keys().at( 0 ).toInt();
	}
	else
	{
		majorityLabel = keysOfLabelGroups().keys().at( 1 ).toInt();
	}

	return majorityLabel;
}

//-----------------------------------------------------------------------------

QStringList DataPackage::getMinorityKeys() const
{
	auto minorityIndex = getMinorityIndex();
	auto key           = keysOfLabelGroups().keys().at( minorityIndex );

	return keysOfLabelGroups().value( key );
}

//-----------------------------------------------------------------------------

QStringList DataPackage::getMajorityKeys() const
{
	auto majorityIndex = getMajorityIndex();
	auto key           = keysOfLabelGroups().keys().at( majorityIndex );

	return keysOfLabelGroups().value( key );

}

//-----------------------------------------------------------------------------

void DataPackage::updateLDB()
{
	lpmldata::TabularData updatedLDB;

	mLDB = labelDatabaseSubset( mFDB.keys() );	
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataPackage::normalizeData()
{
	QVector< QVector< double > > normalizedFeatureColumns;

	auto featureNames = mFDB.headerNames();
	
	for ( auto& featureName : featureNames )
	{
		auto featureColumn           = this->featureColumn( featureName );
		auto normalizedFeatureColumn = this->normalizeFeature( featureColumn );

		normalizedFeatureColumns.push_back( normalizedFeatureColumn );
	}

	auto normalizedFDB = this->featureDatabaseSubset( normalizedFeatureColumns );

	return normalizedFDB;
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage DataPackage::normalizeDataPackage()
{
	auto normalizedFDB = normalizeData();

	return lpmldata::DataPackage( normalizedFDB, mLDB );
}

//-----------------------------------------------------------------------------

}