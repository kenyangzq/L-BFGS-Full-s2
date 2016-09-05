
//
//
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include "./include/cppoptlib/meta.h"
#include "./include/cppoptlib/problem.h"
#include "./include/cppoptlib/lbfgssolver.h"
#include <cstdlib>

#define PI 3.1415926

using namespace std;

// we define a new problem
// we use a templated-class rather than "auto"-lambda function for a clean architecture



void openFile (ifstream & inputfile, string name){
    inputfile.open(name.c_str());
    if (inputfile.fail()) {
        cout << "Error opening input data file\n";
        exit(1);
    }
}

string ParseControlFile(ifstream & inputfile, int & dim, int &numpts, double & s, int & c, int & max_neighbor, int & numFile, int & numIteration, bool & infile){
    
    
    infile = false;
    string filename = "";
    
    string line;
    int lineNumber = 0;
    while (! inputfile.eof()) {
        lineNumber++;
        getline(inputfile, line);
        stringstream tmp(line);
        string k;
        if (lineNumber == 4) {
            tmp >> k >> k >> s;
        }else if (lineNumber == 5) {
            tmp >> k >> k >> dim;
        }else if (lineNumber == 6) {
            tmp >> k >> k >> c;
        }else if (lineNumber == 7) {
            tmp >> k >> k >> infile;
        }else if (lineNumber == 8) {
            tmp >> k >> k >> numpts;
        }else if (lineNumber == 9) {
            tmp >> k >> k >> numIteration;
        }else if (lineNumber == 10) {
            tmp >> k >> k >> numFile;
        }else if (lineNumber == 11) {
            tmp >> k >> k >> max_neighbor;
        }else if (lineNumber == 12) {
            tmp >> k >> k >> filename;
        }
    }
    cout << "\nSummary of the control file:\n\n";
    cout << "S value: " << s << "\n";
    cout << "Dimension: " << dim << "\n";
    cout << "C value: " << c << "\n";
    cout << "Infile request: " << infile << "\n";
    cout << "Number of points: " << numpts << "\n";
    cout << "Number of iterations: " << numIteration << "\n";
    cout << "Number of output files: " << numFile << "\n";
    cout << "Max neighbor: " << max_neighbor << "\n";
    if (infile) cout << "Input filename: " << filename << "\n\n";
    else cout << "No input file request; program will generate a random configuartion.\n\n";
    return filename;
}

double dist_squared(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2)
{
    return (2-2*(sin(angles1(1))*sin(angles2(1))*cos(angles1(0)-angles2(0))+cos(angles1(1))*cos(angles2(1))));
}

void To3D(const Eigen::Matrix<double, 1, 2> & angles, Eigen::Matrix<double, 1, 3> & coords)
{
    coords(0) = cos(angles(0)) * sin(angles(1));
    coords(1) = sin(angles(0)) * sin(angles(1));
    coords(2) = cos(angles(1));
}

void ToVector(const Eigen::MatrixXd & M, cppoptlib::Vector<double> & V )
{
    int c = M.cols();
    for (int i=0; i<M.rows(); ++i )
        V.segment(i*c,c) = M.row(i);
}

void ComputeJacobian(const double & theta, const double & phi, Eigen::Matrix<double, 3, 2> & temp){
    //x = sin(phi) cos(theta)
    //y = sin(phi) sin(theta)
    //z = cos(phi)
    //
    // derivatives w.r.t. theta:
    temp(0,0) = -sin(phi) * sin(theta); // x
    temp(1,0) =  sin(phi) * cos(theta); // y
    temp(2,0) =  0;                      // z
    // derivatives w.r.t. phi:
    temp(0,1) =  cos(phi) * cos(theta); // x
    temp(1,1) =  cos(phi) * sin(theta); // y
    temp(2,1) = -sin(phi);             // z
}


void AngleGradient(const cppoptlib::Vector<double> & all_angles, const int & pt_index, const double & s_power, cppoptlib::Vector<double> & output)
{
    Eigen::Matrix<double, 1, 3> temp_sum, temp_pt, temp_i, temp;
    Eigen::Matrix<double, 3, 2> temp_jacobian;
    temp_sum.setZero();
    for (int i=0; i<pt_index; ++i)
    {
        To3D(all_angles.segment<2>(pt_index*2), temp_pt);
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        double product = temp.dot(temp);
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    for (int i=pt_index+1; i<all_angles.rows()/2; ++i)
    {
        To3D(all_angles.segment<2>(pt_index*2), temp_pt);
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        double product = temp.dot(temp);
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    temp_sum *= -s_power;
    ComputeJacobian(all_angles(pt_index*2+0), all_angles(pt_index*2+1), temp_jacobian);
    output.segment<2>(pt_index*2) = temp_sum * temp_jacobian;
}

void FullGradient(const cppoptlib::Vector<double> & all_angles, const double & s_power, cppoptlib::Vector<double> & output)
{
    for(int i=0; i<all_angles.size()/2; ++i)
    {
        AngleGradient(all_angles, i, s_power, output);
    }
}

double EnergyMatrix(const cppoptlib::Matrix<double> & M, const double s_power)
{
    // M contains spherical coordinates
    double e = 0;
    for (int i=0; i<M.rows(); ++i)
    {
        for (int j=0; j<i; ++j)
        {
            e += pow(dist_squared(M.row(i), M.row(j)), -s_power/2.0);
        }
    }
    return 2.0 * e;
}

double Energy(const cppoptlib::Vector<double> & V, const double s_power, const int dim)
{
    // V contains spherical coordinates
    double e = 0;
    for (int i=0; i<V.size()/dim; ++i)
    {
        for (int j=0; j<i; ++j)
        {
            e += pow(dist_squared(V.segment<2>(i*dim), V.segment<2>(j*dim)), -s_power/2.0);
        }
    }
    return 2.0 * e;
}

void ToAngles(Eigen::MatrixXd & all_points, Eigen::MatrixXd & all_angles)
{
    //cppoptlib::Matrix<double> angles(numpts, dim-1);
    for (int i = 0; i < all_points.rows(); i++) {
        double x = all_points(i,0);
        double y = all_points(i,1);
        double z = all_points(i,2);
        double r = sqrt(x*x+y*y+z*z);
        all_angles(i,0) = atan2(y,x);
        all_angles(i,1) = acos(z/r);
    }
}

class minimizeEnergy : public cppoptlib::Problem<double> {
    double s;
    int dim;
public:
    minimizeEnergy(double s_value, int dim_value):s(s_value),dim(dim_value){};
    
    double value(const cppoptlib::Vector<double> &x) {
        return Energy(x, s, dim-1);
    }
    

    void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad) {
        FullGradient(x, s, grad);
    }
};






void writeFile (ofstream & outputfile, string name, cppoptlib::Vector<double> V, int dim){
    outputfile.open(name.c_str());
    if (outputfile.fail()) {
        cout << "Error writing output data file" << endl;
        exit(1);
    }
    int c = dim-1;
    
    outputfile << setprecision(6);
    outputfile << fixed;
    
    for (int i =0; i < V.rows()/c; i++) {
        cppoptlib::Vector<double> tmp = V.segment(i*c, c);
        Eigen::Matrix<double, 1, 3> tmp2;
        To3D(tmp, tmp2);
        outputfile << tmp2(0) << "\t" << tmp2(1) << "\t" << tmp2(2) << "\n";
    }
}


// generate random sphere configuration
void randptSphere(double coordinates[], int dim){
    
    double z;
    double norm;
    double normsq=2;
    
    while(normsq>1 || normsq==0){
        normsq=0;
        
        for(int i=0;i<dim;i++){
            z=1-(2*(double)rand()/(double)RAND_MAX);
            normsq += z*z;
            coordinates[i] = z;
        }
    }
    
    norm=sqrt(normsq);
    
    for(int i=0;i<dim;i++){
        coordinates[i] = coordinates[i]/norm;
    }
    
}


/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[]) {
    
    
    
    double s;
    int dim = 0, numpts=0, c=0, cubes_per_side=0, max_neighbor=0, numFile=0, numIteration=0;
    ifstream inputfile, pointfile;
    ofstream outputfile;
    bool infile;

    
    openFile(inputfile, "control.inp");
    string filename = ParseControlFile(inputfile, dim, numpts, s, c, max_neighbor, numFile, numIteration, infile);
    inputfile.close();
    openFile(pointfile, filename);
    Eigen::MatrixXd X(numpts, dim), A(numpts, dim-1);
    cppoptlib::Vector<double> V(A.size()), G(A.size());
    
    // read points
    if (infile)
    {
        openFile(pointfile, filename);
        int lineNumber = 0;
        while (!pointfile.eof() && lineNumber < numpts)
        {
            for (int i=0; i<3;  ++i) pointfile >> X(lineNumber, i);
            lineNumber++;
        }
        pointfile.close();
    }
    // generate random configuration
    else
    {
        srand(time(0));
        double apoint[3];
        for (int i = 0; i < numpts; i++) {
            randptSphere(apoint, dim);
            for (int j = 0; j < dim; j++) {
                
                X(i, j) = apoint[j];
            }
        }
    }
    
    
    ToAngles(X,A);
    ToVector(A,V);
    

  
    
    minimizeEnergy f(s, dim);
    cppoptlib::LbfgsSolver<double> solver;
    cout <<"Energy before: " << Energy(V, s, dim-1) << endl;
    
    solver.setNumFile(numFile);
    solver.setNumIteration(numIteration);
    solver.setFileName(filename);
    
    solver.minimize(f, V);
    
    cout << fixed;
    cout << "Energy now: " << f(V) << endl;
    
    writeFile(outputfile, "output.txt", V, dim);
    

    return 0;
    
    
    
}















