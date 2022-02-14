import os
import click

from I3Tray import I3Tray
from icecube import icetray, dataio, hdfwriter
from icecube.weighting import get_weighted_primary

from ic3_labels.labels.modules import MCLabelsCascades
from dnn_reco.ic3.segments import ApplyDNNRecos


@click.command()
@click.argument(
    "input_file_pattern", type=click.Path(exists=True), required=True, nargs=-1
)
@click.option(
    "-o",
    "--outfile",
    default="dnn_output",
    help="Name of output file without file ending.",
)
@click.option(
    "-m",
    "--model_names",
    default="getting_started_model",
    help="Parent directory of exported models.",
)
@click.option(
    "-d",
    "--models_dir",
    default="{DNN_HOME}/exported_models",
    help="Parent directory of exported models.",
)
@click.option(
    "-g",
    "--gcd_file",
    default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2012.56063_V1.i3.gz",
    help="GCD File to use.",
)
@click.option(
    "-j",
    "--num_cpus",
    default=8,
    help="Number of CPUs to use if run on CPU instead of GPU",
)
@click.option("--i3/--no-i3", default=True)
@click.option("--hdf5/--no-hdf5", default=True)
def main(
    input_file_pattern, outfile, model_names, models_dir, gcd_file, num_cpus, i3, hdf5
):

    # create output directory if necessary
    base_path = os.path.dirname(outfile)
    if not os.path.isdir(base_path):
        print("\nCreating directory: {}\n".format(base_path))
        os.makedirs(base_path)

    # expand models_dir with environment variable
    models_dir = models_dir.format(DNN_HOME=os.environ["DNN_HOME"])

    HDF_keys = [
        "LabelsDeepLearning",
        "MCPrimary",
        "OnlineL2_PoleL2MPEFit_MuEx",
        "OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllBINS_Muon",
        "MPEFitFitParams",
        "MPEFit",
    ]

    tray = I3Tray()

    # read in files
    file_name_list = [str(gcd_file)]
    file_name_list.extend(list(input_file_pattern))
    tray.AddModule("I3Reader", "reader", Filenamelist=file_name_list)

    # Add labels
    tray.AddModule(
        get_weighted_primary, "getWeightedPrimary", If=lambda f: not f.Has("MCPrimary")
    )
    tray.AddModule(
        MCLabelsCascades,
        "MCLabelsCascades",
        PulseMapString="InIceDSTPulses",
        PrimaryKey="MCPrimary",
        ExtendBoundary=0.0,
        OutputKey="LabelsDeepLearning",
    )

    # collect model and output names
    if isinstance(model_names, str):
        model_names = [str(model_names)]
    output_names = ["DeepLearningReco_{}".format(m) for m in model_names]

    # Make sure DNN reco will be writen to hdf5 file
    for outbox in output_names:
        if outbox not in HDF_keys:
            HDF_keys.append(outbox)
            HDF_keys.append(outbox + "_I3Particle")

    # Apply DNN Reco
    tray.AddSegment(
        ApplyDNNRecos,
        "ApplyDNNRecos",
        pulse_key="InIceDSTPulses",
        model_names=model_names,
        output_keys=output_names,
        models_dir=models_dir,
        num_cpus=num_cpus,
        dom_exclusions=["SaturationWindows", "BadDomsList", "CalibrationErrata"],
        partial_exclusion=True,
    )

    # Write output
    if i3:
        tray.AddModule("I3Writer", "EventWriter", filename="{}.i3.bz2".format(outfile))

    if hdf5:
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            "hdf",
            Output="{}.hdf5".format(outfile),
            CompressionLevel=9,
            Keys=HDF_keys,
            SubEventStreams=["InIceSplit"],
        )
    tray.AddModule("TrashCan", "YesWeCan")
    tray.Execute()


if __name__ == "__main__":
    main()
