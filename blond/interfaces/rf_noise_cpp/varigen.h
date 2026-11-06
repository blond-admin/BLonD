// hardcopy from https://gitlab.cern.ch/bbielaws/varinoisegen (30.10.2024)
// Commit SHA 9f32cdbffcdaed60e07d3ba3cdcc71b617dede82
#ifndef VARIGEN_DEF
#define VARIGEN_DEF

#include <vector>
#include <stdexcept>

struct VarinoiseParams
{
    int nSource = 2048;
    int nPntMin = 8;
    int rSeed = 0;
    double samplingRate = 11245.49; //Hz
    double rms = 1.0;
};


#ifndef NOISE_HPP
class GenericShapeFunc
{
    public:
        virtual ~GenericShapeFunc() {};
        virtual double get(double x) = 0;
};
#endif //NOISE_HPP

class ConstantShapeFunction: public GenericShapeFunc
{
public:

    ConstantShapeFunction(double x): x(x) {}
    virtual ~ConstantShapeFunction() {}
    virtual double get(double x) {return this->x;}
    double x = 0.0;
};

class DiscreteShapeFunction: public GenericShapeFunc
{
public:
    DiscreteShapeFunction(std::vector<double>& xs, std::vector<double>& ys):
        xs(xs), ys(ys)
    {
        if (xs.size() != ys.size())
            throw std::length_error("Sizes don't match");
    }

    virtual ~DiscreteShapeFunction() {}

    size_t getFirstLargerX(double x)
    {
        size_t size = xs.size();
        size_t firstLargerX = 0;
        for (; firstLargerX < size; firstLargerX++)
        {
            if (xs[firstLargerX] >= x)
                break;
        }
        return firstLargerX;
    }

    virtual double get(double x)
    {
        size_t size = xs.size();
        if (x <= xs[0])
            return ys[0];
        if (x >= xs[size-1])
            return ys[size-1];

        size_t after = getFirstLargerX(x);
        size_t before = after - 1;

        double y0 = ys[before];
        double xSection = x - xs[before];

        double slope = (ys[after] - ys[before]) / (xs[after] - xs[before]);

        return y0 + xSection * slope;
    }

    std::vector<double>& xs;
    std::vector<double>& ys;
};


extern "C" std::vector<double> generateVarinoise(const VarinoiseParams& params,
                    const std::vector<double>& fHigh,
                    const std::vector<double>& fLow,
                    GenericShapeFunc& shapeFunction);

#endif //VARIGEN_DEF
