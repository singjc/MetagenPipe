#!/bin/bash
mkdir humann3_db
humann_databases --download chocophlan full humann3_db --update-config yes
humann_databases --download uniref uniref90_diamond humann3_db --update-config yes
humann_databases --download utility_mapping full humann3_db --update-config yes
