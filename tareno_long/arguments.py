from argparse import ArgumentParser
from datetime import datetime, timedelta

today = datetime.utcnow()
parser = ArgumentParser()


parser.add_argument(
    '--start-date',
    dest='start_date',
    default=today.strftime('%Y-%m-%d'),
    help='The start date to run the calculations')
parser.add_argument(
    '--end-date',
    dest='end_date',
    default=today.strftime('%Y-%m-%d'),
    help='The end date to run the calculations')
parser.add_argument(
    '--load-cutoff',
    dest='load_cutoff',
    default=str(today),
    help='Load cutoff timestamp for loading inputs')
parser.add_argument(
    '--specific-assets',
    dest='specific_assets',
    default='')

validation_override = parser.add_mutually_exclusive_group(required=False)
validation_override.add_argument('--validation', dest='validation_override', action='store_false')
validation_override.add_argument('--no-validation', dest='validation_override', action='store_true')
parser.set_defaults(validation_override=False)


args = parser.parse_args()
