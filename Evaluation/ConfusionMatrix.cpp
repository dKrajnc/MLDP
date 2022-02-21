
#include <Evaluation/ConfusionMatrix.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

ConfusionMatrix::ConfusionMatrix( unsigned int aClassifierCount )
:
    lpmldata::Array2D< unsigned int >( aClassifierCount, aClassifierCount ),
	mAvgTP( 0.0 ),
	mAvgTN( 0.0 ),
	mAvgFP( 0.0 ),
	mAvgFN( 0.0 ),
	mAvgTPR( 0.0 ),
	mAvgFPR( 0.0 ),
	mRoc( 0.0 ),
	mNpv( 0.0 ),
	mAccuracy( 0.0 ),
	mAccuracyLocalAvg( 0.0 ),
	mErrorValues( ),
	mUser( 0.0)
{
}

//-----------------------------------------------------------------------------

ConfusionMatrix::~ConfusionMatrix()
{
	mErrorValues.clear();
}

//-----------------------------------------------------------------------------

void ConfusionMatrix::evaluate()
{
	//initValues();  //Initialize all member variables.
	mRoc = 0.0;

	double diagonalSum = 0.0;
	double accuracyLocalSum = 0.0;
	double rowSum = 0.0;
	double columnSum = 0.0;
	double totalSum = 0.0;
	double TP = 0.0;
	double TN = 0.0;
	double FP = 0.0;
	double FN = 0.0;
	double actualRocDistance = 0.0;
	double actualNpv = 0.0;
	double geomMeanRocDistances = 0.0;
	double geomMeanNpv = 0.0;

	mErrorValues.clear();

	//If it is a classic binary classifier.
	if ( rowCount() == 2 )
	{
		TN = double( this->operator()( 0, 0 ) );
		FN = double( this->operator()( 0, 1 ) );
		FP = double( this->operator()( 1, 0 ) );
		TP = double( this->operator()( 1, 1 ) );

		//Store global values.
		//mAvgTP = TP;
		//mAvgTN = TN;
		//mAvgFP = FP;
		//mAvgFN = FN;

		mRoc = rocDistance( TP, TN, FP, FN );  //This calculates TPR and FPR
		mNpv = ( TN / ( TN + FN ) );// +( TN / ( TN + FP ) );
		//double ratio = ( TP + FN ) / ( FP + TN );
		//mUser = FN / ( ( ( 1.0 - ratio ) * ( TP ) ) + FN + ( ratio * TN ) );

		//mUser = ( TN / ( TN + FP ) ) + ( TN + ( TN + FN ) );
		//mAccuracy = ( TN + TP ) / ( TN + TP + FN + FP );
		//mAccuracyLocalAvg = mAccuracy;
	}
	else  //If it is an unary classifier.
	{
		//Calculate diagonal sum.
		for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
		{
			diagonalSum += this->operator()( rowIndex, rowIndex );
		}

		QList< double > TPRs;

		//Calculate values
		for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
		{
			rowSum = 0.0;
			columnSum = 0.0;

			//Calculate row and column sums.
			for ( unsigned int columnIndex = 0; columnIndex < columnCount(); ++columnIndex )
			{
				rowSum += double( this->operator()( rowIndex, columnIndex ) );  //Sum row (FP + TP)
				columnSum += double( this->operator()( columnIndex, rowIndex ) );  //Sum column (FN + TP)
			}

			totalSum += rowSum;  //Calculate the total sum of the matrix.

			TP = double( this->operator()( rowIndex, rowIndex ) );  //TP is the actual diagonal element.
			FP = rowSum - TP;  //Row contains TP too.
			FN = columnSum - TP;  //Column contains TP too.
			TN = diagonalSum - TP;  //Diagonal sum contains TP too.
			//accuracyLocalSum += TP / rowSum;  //Add local accuracy to global one.

			//TPRs.push_back( TP / columnSum ); // TPR = TP / ( TP + FN )

			//Store to global values.
			/*mAvgTP += TP;
			mAvgTN += TN;
			mAvgFP += FP;
			mAvgFN += FN;*/

			actualRocDistance = rocDistance( TP, TN, FP, FN );  //Calculate the actual ROC distance of the given row/column. This calculates TPR and FPR
			actualNpv = TN / ( TN + FN );
			geomMeanRocDistances += actualRocDistance * actualRocDistance;
			geomMeanNpv += actualNpv * actualNpv;
		}

		//double rowCnt = double( rowCount() );

		//Fix global values to hold averages.
		//mAvgTP /= rowCnt;
		//mAvgTN /= rowCnt;
		//mAvgFP /= rowCnt;
		//mAvgFN /= rowCnt;
		//mAvgTPR /= rowCnt;
		//mAvgFPR /= rowCnt;
		mRoc = sqrt( geomMeanRocDistances );
		mNpv = sqrt( geomMeanNpv );

		//mAccuracy = diagonalSum / totalSum;
		//mAccuracyLocalAvg = accuracyLocalSum / rowCnt;
	}
}

//-----------------------------------------------------------------------------

//double ConfusionMatrix::UWF( double aRatio )
//{
//	initValues();  //Initialize all member variables.
//
//	double UWF = 0.0;
//	double betaSqr = std::pow( aRatio, 2 );
//
//	double rowSum = 0.0;
//	double columnSum = 0.0;
//	double totalSum = 0.0;
//	double diagonalSum = 0.0;
//	double TP = 0.0;
//	double TN = 0.0;
//	double FP = 0.0;
//	double FN = 0.0;
//	QVector< double > labelOccurrences;
//	labelOccurrences.resize( rowCount() );
//
//	mErrorValues.clear();
//
//	//Calculate the total number of entries
//	for ( unsigned int columnIndex = 0; columnIndex < columnCount(); ++columnIndex )
//	{
//		diagonalSum += this->operator()( columnIndex, columnIndex );
//
//		for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
//		{
//			double entry = double( this->operator()( rowIndex, columnIndex ) );
//			totalSum += entry;
//			labelOccurrences[ columnIndex ] += entry;
//		}
//	}
//
//	for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
//	{
//		labelOccurrences[ rowIndex ] /= totalSum;
//	}
//
//	//If it is a classic binary classifier.
//	if ( rowCount() == 2 )
//	{
//		TN = double( this->operator()( 0, 0 ) );
//		FN = double( this->operator()( 0, 1 ) );
//		FP = double( this->operator()( 1, 0 ) );
//		TP = double( this->operator()( 1, 1 ) );
//
//		double TPR = TP / ( TP + FN );  //True Positive Rate.
//		double FPR = FP / ( FP + TN );  //False Positive Rate.
//		double a = ( FPR * TPR ) / 2.0;
//		double b = ( 1.0 - FPR ) * TPR;
//		double c = ( ( 1.0 - FPR ) * ( 1.0 - TPR ) ) / 2.0;
//		double AUC = a + b + c;
//		//UWF += ( 1.0 - labelOccurrences[ 0 ] ) * ( ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP ) );
//		UWF += ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP );
//
//		QList< double > errors = { TPR, 1.0 - FPR, AUC, UWF, TP, TN, FP, FN };
//		mErrorValues.push_back( errors );
//
//		TN = double( this->operator()( 1, 1 ) );
//		FN = double( this->operator()( 1, 0 ) );
//		FP = double( this->operator()( 0, 1 ) );
//		TP = double( this->operator()( 0, 0 ) );
//
//		//UWF += ( 1.0 - labelOccurrences[ 1 ] ) * ( ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP ) );
//		UWF += ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP );
//	}
//	else  //If it is an unary classifier.
//	{
//		for ( unsigned int rowIndex = 0; rowIndex < rowCount(); ++rowIndex )
//		{
//			rowSum = 0.0;
//			columnSum = 0.0;
//
//			//Calculate row and column sums.
//			for ( unsigned int columnIndex = 0; columnIndex < columnCount(); ++columnIndex )
//			{
//				rowSum += double( this->operator()( rowIndex, columnIndex ) );  //Sum row (FP + TP)
//				columnSum += double( this->operator()( columnIndex, rowIndex ) );  //Sum column (FN + TP)
//			}
//
//			TP = double( this->operator()( rowIndex, rowIndex ) );  //TP is the actual diagonal element.
//			FP = rowSum - TP;  //Row contains TP too.
//			FN = columnSum - TP;  //Column contains TP too.
//			TN = diagonalSum - TP;  //Diagonal sum contains TP too.
//
//			double TPR = TP / ( TP + FN );  //True Positive Rate.
//			double FPR = FP / ( FP + TN );  //False Positive Rate.
//			double a = ( FPR * TPR ) / 2.0;
//			double b = ( 1.0 - FPR ) * TPR;
//			double c = ( ( 1.0 - FPR ) * ( 1.0 - TPR ) ) / 2.0;
//			double AUC = a + b + c;
//			//UWF += ( 1.0 - labelOccurrences[ rowIndex ] ) * ( ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP ) );
//			UWF += ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP );
//
//			QList< double > errors = { TPR, 1.0 - FPR, AUC, UWF, TP, TN, FP, FN };
//			mErrorValues.push_back( errors );
//		}
//	}
//
//	UWF /= rowCount();
//
//	return UWF;
//}

//-----------------------------------------------------------------------------

void ConfusionMatrix::initValues()
{
	mAvgTP = 0.0;
	mAvgTN = 0.0;
	mAvgFP = 0.0;
	mAvgFN = 0.0;
	mAvgTPR = 0.0;
	mAvgFPR = 0.0;
	mRoc = 0.0;
	mAccuracy = 0.0;
	mAccuracyLocalAvg = 0.0;
}

//-----------------------------------------------------------------------------

double ConfusionMatrix::rocDistance( double aTP, double aTN, double aFP, double aFN )
{
	double TPR = aTP / ( aTP + aFN );  //True Positive Rate.
	double FPR = aFP / ( aFP + aTN );  //False Positive Rate.

	double a = ( FPR * TPR ) / 2.0;
	double b = ( 1.0 - FPR ) * TPR;
	double c = ( ( 1.0 - FPR ) * ( 1.0 - TPR ) ) / 2.0;
	double AUC = a + b + c;

	QList< double > errors = { TPR, 1.0 - FPR, AUC, aTP, aTN, aFP, aFN };
	mErrorValues.push_back( errors );

	//Store global calculations.
	//mAvgTPR += TPR;
	//mAvgFPR += FPR;

	//Return with the ROC distance.
	return sqrt( ( ( 1.0 - TPR ) * ( 1.0 - TPR ) ) + ( FPR *  FPR ) );
}

//-----------------------------------------------------------------------------

double ConfusionMatrix::rocDistance2( const QList< double >& aTPRs )
{
	double roc = 0.0;

	for ( int groupIndex = 0; groupIndex < aTPRs.size(); ++groupIndex )
	{
		roc += ( 1.0 - aTPRs.at( groupIndex ) ) * ( 1.0 - aTPRs.at( groupIndex ) );
	}

	return sqrt( roc );
}

//-----------------------------------------------------------------------------

}
